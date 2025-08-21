import sys
import yaml
import os

import torch
import numpy as np

import jax
import jax.numpy as jnp
from flax import serialization
from torchvision.utils import save_image
from unet import UNet
from dit import DiT
from sampler import DDPMSampler
from model import VQModel


def load_checkpoint(path, state_template):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return serialization.from_bytes(state_template, f.read())


def generate_samples(rng, model, model_params, sampler, shape, num_steps):
    def make_predict_fn(*, apply_fn, sampler):
        def predict_fn(params, rng, xt, t, t_prev):
            predicted_noise = apply_fn(params, xt, t, None)
            xt_prev = sampler.remove_noise(rng, xt, predicted_noise, t, t_prev)
            return xt_prev

        return jax.pmap(predict_fn, axis_name='batch', donate_argnums=())

    def shard(x):
        n, *s = x.shape
        return x.reshape((num_devices, n // num_devices, *s))

    def unshard(x):
        d, b, *s = x.shape
        return x.reshape((d * b, *s))

    devices = jax.local_devices()
    num_devices = len(devices)
    replicate = lambda tree: jax.device_put_replicated(tree, devices)
    unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)

    predict_fn = make_predict_fn(apply_fn=model.apply, sampler=sampler)

    params_repl = replicate(model_params)
    rng, sample_rng = jax.random.split(rng, 2)

    xt = jax.random.normal(sample_rng, shape=shape)
    timesteps = jnp.arange(0, sampler.total_timesteps, sampler.total_timesteps // num_steps)[::-1]
    timesteps_prev = jnp.concatenate([timesteps[1:], jnp.array([0], dtype=jnp.int32)], axis=0)

    for i in range(len(timesteps)):
        rng, sample_rng = jax.random.split(rng, 2)
        t = jnp.full((shape[0],), timesteps[i], dtype=jnp.int32)
        t_prev = jnp.full((shape[0],), timesteps_prev[i], dtype=jnp.int32)

        xt = jax.tree_util.tree_map(lambda x: shard(x), xt)
        t = jax.tree_util.tree_map(lambda x: shard(x), t)
        t_prev = jax.tree_util.tree_map(lambda x: shard(x), t_prev)
        rng_shard = jax.random.split(sample_rng, num_devices)

        xt = predict_fn(params_repl, rng_shard, xt, t, t_prev)
        xt = jax.tree_util.tree_map(lambda x: unshard(x), xt)

    return xt


def main(config_path):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['model']
    vqmodel_config = config['vqmodel']
    sampler_config = config['sampler']

    seed = diffusion_config['seed']
    key = jax.random.PRNGKey(seed)

    if diffusion_config['target'] == 'DiT':
        diffusion = DiT(**diffusion_config['params'])
    elif diffusion_config['target'] == 'UNet':
        diffusion = UNet(**diffusion_config['params'])
    else:
        raise 'There is no such model'

    vqmodel = VQModel(**vqmodel_config['params'])
    sampler = DDPMSampler(**sampler_config['params'])

    checkpoint_path = diffusion_config['checkpoint_path']
    vqmodel_checkpoint_path = vqmodel_config['checkpoint_path']
    diffusion_params = load_checkpoint(checkpoint_path, None)['ema_params']
    vqmodel_params = load_checkpoint(vqmodel_checkpoint_path, None)['ema_params']

    latents = generate_samples(key, diffusion, diffusion_params, sampler, (64, 16, 16, 4), sampler.total_timesteps)

    x_gen = vqmodel.apply(vqmodel_params, latents, method=vqmodel.decode)

    x_gen = (x_gen + 1.0) / 2.0
    x_gen = jnp.clip(x_gen, 0.0, 1.0)

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        save_image(img, f'/content/drive/MyDrive/Stable-Diffusion/gen_images/generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1])
