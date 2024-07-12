# jit-compatible RGB observations. Currently experimental!
# if it proves useful and necessary in the future, I will consider rewriting env.render in such style also
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

from ..benchmarks import load_bz2_pickle, save_bz2_pickle
from ..core.constants import NUM_COLORS, NUM_LAYERS, TILES_REGISTRY
from ..rendering.rgb_render import render_tile
from ..wrappers import Wrapper

CACHE_PATH = os.environ.get("XLAND_MINIGRID_CACHE", os.path.expanduser("~/.xland_minigrid"))
FORCE_RELOAD = os.environ.get("XLAND_MINIGRID_RELOAD_CACHE", False)


def build_cache(tiles: np.ndarray, tile_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    cache = np.zeros((tiles.shape[0], tiles.shape[1], tile_size, tile_size, 3), dtype=np.uint8)
    agent_cache = np.zeros((tiles.shape[0], tiles.shape[1], tile_size, tile_size, 3), dtype=np.uint8)

    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            # rendering tile
            tile_img = render_tile(
                tile=tuple(tiles[y, x]),
                agent_direction=None,
                highlight=False,
                tile_size=int(tile_size),
            )
            cache[y, x] = tile_img

            # rendering agent on top
            tile_w_agent_img = render_tile(
                tile=tuple(tiles[y, x]),
                agent_direction=0,
                highlight=False,
                tile_size=int(tile_size),
            )
            agent_cache[y, x] = tile_w_agent_img

    return cache, agent_cache


# building cache of pre-rendered tiles
TILE_SIZE = 32

cache_path = os.path.join(CACHE_PATH, "render_cache")

if not os.path.exists(cache_path) or FORCE_RELOAD:
    os.makedirs(CACHE_PATH, exist_ok=True)
    print("Building rendering cache, may take a while...")
    TILE_CACHE, TILE_W_AGENT_CACHE = build_cache(np.asarray(TILES_REGISTRY), tile_size=TILE_SIZE)
    TILE_CACHE = jnp.asarray(TILE_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)
    TILE_W_AGENT_CACHE = jnp.asarray(TILE_W_AGENT_CACHE).reshape(-1, TILE_SIZE, TILE_SIZE, 3)

    print(f"Done. Cache is saved to {cache_path} and will be reused on consequent runs.")
    save_bz2_pickle({"tile_cache": TILE_CACHE, "tile_agent_cache": TILE_W_AGENT_CACHE}, cache_path)

TILE_CACHE = load_bz2_pickle(cache_path)["tile_cache"]
TILE_W_AGENT_CACHE = load_bz2_pickle(cache_path)["tile_agent_cache"]


# rendering with cached tiles
def _render_obs(obs: jax.Array) -> jax.Array:
    view_size = obs.shape[0]

    obs_flat_idxs = obs[:, :, 0] * NUM_COLORS + obs[:, :, 1]
    # render all tiles
    rendered_obs = jnp.take(TILE_CACHE, obs_flat_idxs, axis=0)

    # add agent tile
    agent_tile = TILE_W_AGENT_CACHE[obs_flat_idxs[view_size - 1, view_size // 2]]
    rendered_obs = rendered_obs.at[view_size - 1, view_size // 2].set(agent_tile)
    # [view_size, view_size, tile_size, tile_size, 3] -> [view_size * tile_size, view_size * tile_size, 3]
    rendered_obs = rendered_obs.transpose((0, 2, 1, 3, 4)).reshape(view_size * TILE_SIZE, view_size * TILE_SIZE, 3)

    return rendered_obs


class RGBImgObservationWrapper(Wrapper):
    def observation_shape(self, params):
        new_shape = (params.view_size * TILE_SIZE, params.view_size * TILE_SIZE, 3)

        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"img": new_shape}}
        else:
            obs_shape = new_shape

        return obs_shape

    def __convert_obs(self, timestep):
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            rendered_obs = {**timestep.observation, **{"img": _render_obs(timestep.observation["img"])}}
        else:
            rendered_obs = _render_obs(timestep.observation)

        timestep = timestep.replace(observation=rendered_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__convert_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__convert_obs(timestep)
        return timestep
