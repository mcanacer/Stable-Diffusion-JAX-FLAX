import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence


def truncated_normal(stddev, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        return jax.random.truncated_normal(
            key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
    return init


class DiscriminatorBlock(nn.Module):
    in_channels: int
    out_channels: int
    downsample: bool = True

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), kernel_init=truncated_normal(0.02))(x)

        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), kernel_init=truncated_normal(0.02))(x)

        if self.in_channels != self.out_channels:
            residual = nn.Conv(features=self.out_channels, kernel_size=(1, 1), kernel_init=truncated_normal(0.02))(residual)

        x = x + residual

        if self.downsample:
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        return x


class Discriminator(nn.Module):
    channel_multipliers: Sequence[int]
    base_channels: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.base_channels, kernel_size=(3, 3), kernel_init=truncated_normal(0.02))(x)

        channels = self.base_channels
        for mult in self.channel_multipliers:
            out_channels = self.base_channels * mult
            x = DiscriminatorBlock(channels, out_channels)(x)
            channels = out_channels

        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.swish(x)
        x = nn.Conv(1, kernel_size=(3, 3), kernel_init=truncated_normal(0.02))(x)  # [N, H', W', 1] → patch-level logits

        return x


if __name__ == '__main__':
    disc = Discriminator([1, 2, 4, 8])
    disc_vars = disc.init(jax.random.PRNGKey(0), jnp.ones((2, 256, 256, 3)))
    logits = disc.apply(disc_vars, jnp.ones((2, 256, 256, 3), dtype=jnp.float32))
