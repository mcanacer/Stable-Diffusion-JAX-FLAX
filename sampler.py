import jax
import jax.numpy as jnp

import math


class DDPMSampler(object):
    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type="linear"):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.beta = jnp.linspace(beta_start, beta_end, total_timesteps, dtype=jnp.float32)  # [T]
        elif schedule_type == "cosine":
            self.beta = self._cosine_beta_schedule(total_timesteps)  # [T]
        else:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        self.alpha = 1.0 - self.beta  # [T]
        self.alpha_cum_prod = jnp.cumprod(self.alpha)  # [T]

    def _cosine_beta_schedule(self, timesteps, s=0.008, max_beta=0.999):
        def alpha_bar_fn(t):
            return math.cos(((t / timesteps + s) / (1 + s)) * math.pi / 2) ** 2

        betas = []
        for t in range(timesteps):
            t1 = t / timesteps
            t2 = (t + 1) / timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return jnp.array(betas, dtype=jnp.float32)

    def add_noise(self, rng, x0, timesteps):
        """q(x_t | x_0) forward process"""
        alpha_bar = self.alpha_cum_prod[timesteps]  # [N]
        noise = jax.random.normal(rng, shape=x0.shape)  # [N, H, W, C]

        alpha_bar = jnp.expand_dims(alpha_bar, axis=(1, 2, 3))  # [N, 1, 1, 1]
        noisy_image = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise

        return noisy_image, noise

    def remove_noise(self, rng, xt, predicted_noise, timesteps, timesteps_prev):
        """p(x_{t-1} | x_t) reverse process"""

        beta_t = jnp.expand_dims(self.beta[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_t = jnp.expand_dims(self.alpha[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_bar_t = jnp.expand_dims(self.alpha_cum_prod[timesteps], axis=(1, 2, 3))  # [N, 1, 1, 1]
        alpha_bar_prev_t = jnp.expand_dims(self.alpha_cum_prod[timesteps_prev], axis=(1, 2, 3))  # [N, 1, 1, 1]

        # Compute posterior mean
        coef1 = 1.0 / jnp.sqrt(alpha_t)
        coef2 = beta_t / jnp.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (xt - coef2 * predicted_noise)

        # Compute posterior variance
        var = beta_t * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        log_var = jnp.log(jnp.clip(var, a_min=1e-20))

        # Sample noise
        noise = jax.random.normal(rng, shape=xt.shape)

        # If t == 0, just return mean
        denoised = mean + jnp.exp(0.5 * log_var) * noise
        denoised = jnp.where(jnp.expand_dims(timesteps, axis=(1, 2, 3)) == 0, mean, denoised)

        return denoised


class DDIMSampler(object):
    def __init__(self, total_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type="linear"):
        self.total_timesteps = total_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.beta = jnp.linspace(beta_start, beta_end, total_timesteps, dtype=jnp.float32)  # [T]
        elif schedule_type == "cosine":
            self.beta = self._cosine_beta_schedule(total_timesteps)
        else:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")

        self.alpha = 1.0 - self.beta  # [T]
        self.alpha_cum_prod = jnp.cumprod(self.alpha)  # [T]

    def _cosine_beta_schedule(self, timesteps, s=0.008, max_beta=0.999):
        def alpha_bar_fn(t):
            return math.cos(((t / timesteps + s) / (1 + s)) * math.pi / 2) ** 2

        betas = []
        for t in range(timesteps):
            t1 = t / timesteps
            t2 = (t + 1) / timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return jnp.array(betas, dtype=jnp.float32)

    def add_noise(self, rng, x0, timesteps):
        """Forward process q(x_t | x_0)"""
        alpha_bar = self.alpha_cum_prod[timesteps]  # [N]
        noise = jax.random.normal(rng, shape=x0.shape)  # [N, H, W, C]

        alpha_bar = jnp.expand_dims(alpha_bar, axis=(1, 2, 3))  # [N, 1, 1, 1]
        noisy_image = jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise

        return noisy_image, noise

    def remove_noise(self, rng, xt, predicted_noise, timesteps, timesteps_prev, eta=0.0):
        """DDIM reverse process p(x_{t-1} | x_t, x0)"""
        alpha_bar_t = jnp.expand_dims(self.alpha_cum_prod[timesteps], axis=(1, 2, 3))
        alpha_bar_prev_t = jnp.expand_dims(self.alpha_cum_prod[timesteps_prev], axis=(1, 2, 3))

        sigma_t = eta * jnp.sqrt(
            (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) *
            (1.0 - alpha_bar_t / alpha_bar_prev_t)
        )
        noise = jax.random.normal(rng, shape=xt.shape)

        x0_pred = (xt - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise) / jnp.sqrt(alpha_bar_t)

        dir_xt = jnp.sqrt(1.0 - alpha_bar_prev_t - sigma_t ** 2) * predicted_noise
        x_prev_t = jnp.sqrt(alpha_bar_prev_t) * x0_pred + dir_xt + sigma_t * noise

        return x_prev_t
