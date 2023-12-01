import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import room, sample_coordinates, sample_direction
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


class Empty(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=9, width=9)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 4 * (params.height * params.width)

    def _generate_problem(self, params: EnvParams, key: jax.Array):
        grid = room(params.height, params.width)

        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
        agent = AgentState(position=jnp.array((1, 1)), direction=jnp.asarray(1))

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


class EmptyRandom(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=9, width=9)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 4 * (params.height * params.width)

    def _generate_problem(self, params: EnvParams, key: jax.Array):
        key, pos_key, dir_key = jax.random.split(key, num=3)

        grid = room(params.height, params.width)
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        agent = AgentState(
            position=sample_coordinates(pos_key, grid, num=1).squeeze(), direction=sample_direction(dir_key)
        )
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
