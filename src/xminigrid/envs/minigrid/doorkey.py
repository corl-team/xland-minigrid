from __future__ import annotations

import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import coordinates_mask, room, sample_coordinates, sample_direction, vertical_line
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


class DoorKey(Environment[EnvParams, EnvCarry]):
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=5, width=5)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=10 * (params.height * params.width))
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        key, _key = jax.random.split(key)
        keys = jax.random.split(_key, num=4)

        wall_pos = jax.random.randint(keys[0], shape=(), minval=2, maxval=params.width - 2)
        door_pos = jax.random.randint(keys[1], shape=(), minval=1, maxval=params.width - 2)

        grid = room(params.height, params.width)
        grid = vertical_line(grid, wall_pos, 0, params.height, tile=TILES_REGISTRY[Tiles.WALL, Colors.GREY])
        grid = grid.at[door_pos, wall_pos].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, Colors.YELLOW])
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        # mask out positions after the wall, so that agent and key are always on the opposite side from the goal
        # WARN: this is a bit expensive, judging by the FPS benchmark
        mask = coordinates_mask(grid, (params.height, wall_pos), comparison_fn=jnp.less)
        key_coords, agent_coords = sample_coordinates(keys[2], grid, num=2, mask=mask)

        grid = grid.at[key_coords[0], key_coords[1]].set(TILES_REGISTRY[Tiles.KEY, Colors.YELLOW])

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[3]))
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
