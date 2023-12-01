import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import four_rooms, sample_coordinates, sample_direction
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


class FourRooms(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=19, width=19)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        # TODO: this is hardcoded and thus problematic. Move it to EnvParams?
        return 100

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        key, *keys = jax.random.split(key, num=4)

        grid = four_rooms(params.height, params.width)

        black_floor = TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]
        # sampling doors positions
        doors_offsets = jax.random.randint(keys[0], shape=(4,), minval=1, maxval=params.height // 2)
        grid = grid.at[params.height // 2, doors_offsets[0]].set(black_floor)
        grid = grid.at[params.height // 2, params.width // 2 + doors_offsets[1]].set(black_floor)
        grid = grid.at[doors_offsets[2], params.width // 2].set(black_floor)
        grid = grid.at[params.height // 2 + doors_offsets[3], params.width // 2].set(black_floor)

        # sampling agent and goal positions
        goal_coords, agent_coords = sample_coordinates(keys[1], grid, num=2)
        grid = grid.at[goal_coords[0], goal_coords[1]].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[2]))
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
