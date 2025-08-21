import jax
import jax.numpy as jnp
import flax.linen as nn

import math


def modulate(x, shift, scale):
    return x * (1. + scale[:, None]) + shift[:, None]

# From https://github.com/young-geng/m3ae_public/blob/master/m3ae/model.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = jnp.sin(out) # (M, D/2)
    emb_cos = jnp.cos(out) # (M, D/2)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0
    )


def get_2d_sincos_pos_embed(rng, embed_dim, length):
    # example: embed_dim = 256, length = 16*16
    grid_size = int(length ** 0.5)
    assert grid_size * grid_size == length
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0) # (1, H*W, D)


class PatchEmbed(nn.Module):
    hidden_size: int
    patch_size: int

    @nn.compact
    def __call__(self, x):
        N, H, W, C = x.shape
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding='VALID',
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = x.reshape(N, -1, x.shape[-1])
        return x


class TimestepEmbedder(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, timesteps):
        half_dim = self.dim // 2
        emb = -math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * emb)
        emb = timesteps.astype(jnp.float32)[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        emb = nn.Dense(self.dim, kernel_init=nn.initializers.normal(0.02))(emb)
        emb = jax.nn.silu(emb)
        emb = nn.Dense(self.dim, kernel_init=nn.initializers.normal(0.02))(emb)
        return emb


class MLP(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(4 * self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(x)
        return x


class DiTBlocks(nn.Module):
    hidden_size: int
    n_heads: int

    @nn.compact
    def __call__(self, x, c):
        c = jax.nn.silu(c)  # [N, D]
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c)  # [N, 6D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.array_split(c, 6, axis=1)

        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.hidden_size,
            out_features=self.hidden_size,
            kernel_init=nn.initializers.xavier_uniform()
        )(x_modulated, x_modulated)
        x = x + gate_msa[:, None] * attn

        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * MLP(self.hidden_size)(x_modulated)
        return x


class FinalLayer(nn.Module):
    patch_size: int
    hidden_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        c = jax.nn.silu(c)  # [N, D]
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c)  # [N, 2D]
        shift, scale = jnp.array_split(c, 2, axis=1)

        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = modulate(x_norm, shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, kernel_init=nn.initializers.constant(0.))(x)
        return x


class DiT(nn.Module):
    patch_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    out_channels: int

    @nn.compact
    def __call__(self, x, t, c):
        num_patches = (x.shape[1] // self.patch_size) ** 2
        pos_embed = self.param('pos_embed', get_2d_sincos_pos_embed, self.hidden_size, num_patches)  # [1, S, hidden_size]
        pos_embed = jax.lax.stop_gradient(pos_embed)

        x = PatchEmbed(self.hidden_size, self.patch_size)(x) + pos_embed  # [N, S, hidden_size]
        t = TimestepEmbedder(self.hidden_size)(t)  # [N, hidden_size]

        for _ in range(self.num_layers):
            x = DiTBlocks(self.hidden_size, self.num_heads)(x, t)

        x = FinalLayer(self.patch_size, self.hidden_size, self.out_channels)(x, t)
        x = self.unpatchify(x)
        return x

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        x = x.reshape(x.shape[0], h * p, w * p, c)
        return x
