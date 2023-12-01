import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import TileOnPositionGoal
from ...core.grid import coordinates_mask, sample_coordinates, sample_direction, two_rooms
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# colors like in the original minigrid
_allowed_colors = jnp.array(
    (
        Colors.RED,
        Colors.GREEN,
        Colors.BLUE,
        Colors.PURPLE,
        Colors.YELLOW,
        Colors.GREY,
    )
)
_rule_encoding = EmptyRule().encode()[None, ...]


class Unlock(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=6, width=11)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 8 * params.height**2

    def _generate_problem(self, params: EnvParams, key: jax.Array):
        key, *keys = jax.random.split(key, num=5)

        color = jax.random.choice(keys[0], _allowed_colors)
        door_pos = jax.random.randint(keys[1], shape=(), minval=1, maxval=params.height - 1)

        grid = two_rooms(params.height, params.width)
        grid = grid.at[door_pos, params.width // 2].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, color])

        # mask out positions after the wall, so that agent and key are always on the same side
        # WARN: this is a bit expensive, judging by the FPS benchmark
        mask = coordinates_mask(grid, (params.height, params.width // 2), comparison_fn=jnp.less)
        key_coords, agent_coords = sample_coordinates(keys[2], grid, num=2, mask=mask)

        grid = grid.at[key_coords[0], key_coords[1]].set(TILES_REGISTRY[Tiles.KEY, color])

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[3]))
        goal_encoding = TileOnPositionGoal(
            tile=TILES_REGISTRY[Tiles.DOOR_OPEN, color],
            position=jnp.array((door_pos, params.width // 2)),
        ).encode()

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
