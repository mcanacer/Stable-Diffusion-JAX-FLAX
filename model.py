from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


# https://github.com/google-research/maskgit


def entropy_loss(affinity, loss_type="softmax", temperature=1.0):
    """Calculates the entropy loss."""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = jax.nn.softmax(flat_affinity, axis=-1)
    log_probs = jax.nn.log_softmax(flat_affinity + 1e-5, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = jnp.argmax(flat_affinity, axis=-1)
        onehots = jax.nn.one_hot(
            codes, flat_affinity.shape[-1], dtype=flat_affinity.dtype)
        onehots = probs - jax.lax.stop_gradient(probs - onehots)
        target_probs = onehots
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = jnp.mean(target_probs, axis=0)
    avg_entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + 1e-5))
    sample_entropy = -jnp.mean(jnp.sum(target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class ResidualBlock(nn.Module):
    filters: int

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        residual = x
        x = nn.GroupNorm(num_groups=32, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(self.filters, kernel_size=(3, 3))(x)
        x = nn.GroupNorm(num_groups=32, dtype=jnp.float32)(x)
        x = nn.swish(x)

        x = nn.Conv(self.filters, kernel_size=(3, 3), use_bias=False)(x)

        if input_dim != self.filters:
            residual = nn.Conv(self.filters, kernel_size=(1, 1), use_bias=False)(x)

        return x + residual


class AttentionBlock(nn.Module):
    n_heads: int = 4

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        residual = x
        x = x.reshape(B, H * W, C)
        x = nn.LayerNorm()(x)

        x = nn.SelfAttention(
            num_heads=self.n_heads,
            qkv_features=C,
            out_features=C,
            deterministic=True
        )(x)

        x = x.reshape(B, H, W, C)
        return residual + x


class Encoder(nn.Module):
    latent_dim: int
    channel_multipliers: Sequence[int]
    attn_resolutions: Sequence[int]
    filters: int = 128
    num_res_block: int = 2
    n_heads: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.filters, kernel_size=(3, 3), use_bias=False)(x)
        resolution = x.shape[1]
        num_blocks = len(self.channel_multipliers)

        for i in range(num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_block):
                x = ResidualBlock(filters)(x)
                if resolution in self.attn_resolutions:
                    x = AttentionBlock(n_heads=self.n_heads)(x)
            if i < num_blocks - 1:
                x = nn.Conv(features=filters, kernel_size=(4, 4), strides=(2, 2))(x)
                resolution = resolution // 2

        for _ in range(self.num_res_block):
            x = ResidualBlock(filters)(x)

        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(1, 1))(x)

        return x


class Decoder(nn.Module):
    output_channels: int
    channel_multipliers: Sequence[int]
    attn_resolutions: Sequence[int]
    filters: int = 128
    num_res_block: int = 2
    n_heads: int = 1

    @nn.compact
    def __call__(self, x):
        resolution = x.shape[1]  # H = W
        num_blocks = len(self.channel_multipliers)

        filters = self.filters * self.channel_multipliers[-1]
        x = nn.Conv(features=filters, kernel_size=(3, 3))(x)

        for _ in range(self.num_res_block):
            x = ResidualBlock(filters)(x)
            if resolution in self.attn_resolutions:
                x = AttentionBlock(n_heads=self.n_heads)(x)

        for i in reversed(range(num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            for _ in range(self.num_res_block):
                x = ResidualBlock(filters)(x)
                if resolution in self.attn_resolutions:
                    x = AttentionBlock(n_heads=self.n_heads)(x)
            if i > 0:
                n, h, w, c = x.shape
                x = jax.image.resize(x, (n, h * 2, w * 2, c), method='nearest')
                x = nn.Conv(features=filters, kernel_size=(3, 3))(x)
                resolution *= 2

        x = nn.GroupNorm(num_groups=8, dtype=jnp.float32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.output_channels, kernel_size=(3, 3))(x)

        return x


class VectorQuantizer(nn.Module):
    embedding_dim: int
    num_embeddings: int
    commitment_cost: float

    @nn.compact
    def __call__(self, z_e):  # z_e -> [N, H, W, D]
        # Codebook
        codebook = self.param(
            'codebook',
            jax.nn.initializers.variance_scaling(
                scale=1.0, mode='fan_in', distribution='uniform'),
            (self.num_embeddings, self.embedding_dim))
        codebook = jnp.asarray(codebook, dtype=jnp.float32)
        z_e_flat = jnp.reshape(z_e, (-1, self.embedding_dim))  # [NxHxW, D]
        distances = (
            jnp.sum(z_e_flat**2, axis=1, keepdims=True)
            - 2 * jnp.dot(z_e_flat, codebook.T)
            + jnp.sum(codebook**2, axis=1)
        )  # [NxHxW, K]

        encoding_indices = jnp.argmin(distances, axis=1)  # [NxHxW]
        quantized = codebook[encoding_indices]  # [NxHxW, D]

        ent_loss = 0.1 * entropy_loss(-distances, temperature=0.01)

        quantized = jnp.reshape(quantized, z_e.shape)  # [N, H, W, D]
        commitment_loss = self.commitment_cost * jnp.mean((jax.lax.stop_gradient(quantized) - z_e) ** 2)
        embedding_loss = jnp.mean((quantized - jax.lax.stop_gradient(z_e)) ** 2)

        quantized = z_e + jax.lax.stop_gradient(quantized - z_e)
        return quantized, commitment_loss, embedding_loss, ent_loss, encoding_indices


class VQModel(nn.Module):
    channel_multipliers: Sequence[int]
    embedding_dim: int = 256
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    output_channels: int = 3
    attn_resolutions: Sequence[int] = (16,)
    n_heads: int = 1

    def setup(self):
        self.encoder = Encoder(
            latent_dim=self.embedding_dim,
            channel_multipliers=self.channel_multipliers,
            attn_resolutions=self.attn_resolutions,
            n_heads=self.n_heads,
        )
        self.quantizer = VectorQuantizer(
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_embeddings,
            commitment_cost=self.commitment_cost,
        )
        self.decoder = Decoder(
            output_channels=self.output_channels,
            channel_multipliers=self.channel_multipliers,
            attn_resolutions=self.attn_resolutions,
            n_heads=self.n_heads,
        )

        self.quant_conv = nn.Conv(self.embedding_dim, kernel_size=(1, 1))
        self.post_quant_conv = nn.Conv(self.embedding_dim, kernel_size=(1, 1))

    def __call__(self, x):
        encoded = self.encoder(x)
        z_e = self.quant_conv(encoded)
        z_q, commitment_loss, embedding_loss, ent_loss, enc_indices = self.quantizer(z_e)
        z_q = self.post_quant_conv(z_q)
        x_recon = self.decoder(z_q)
        return x_recon, z_q, commitment_loss, embedding_loss, ent_loss, enc_indices

    def encode(self, x):
        encoded = self.encoder(x)  # [N, H, W, D]
        z_e = self.quant_conv(encoded)
        z_q, commitment_loss, embedding_loss, ent_loss, enc_indices = self.quantizer(z_e)
        return z_q  # [N, H, W, D]

    def decode(self, x):
        # x: [N, H, W, D]
        x = self.post_quant_conv(x)
        return self.decoder(x)
