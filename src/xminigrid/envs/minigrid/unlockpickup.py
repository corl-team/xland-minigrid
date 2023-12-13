import jax
import jax.numpy as jnp

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentHoldGoal
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
_allowed_entities = jnp.array(
    (
        Tiles.BALL,
        Tiles.SQUARE,
        Tiles.PYRAMID,
        Tiles.STAR,
        Tiles.HEX,
    )
)
_rule_encoding = EmptyRule().encode()[None, ...]


class UnlockPickUp(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = super().default_params(height=6, width=11)
        default_params = default_params.replace(**kwargs)
        return default_params

    def time_limit(self, params: EnvParams) -> int:
        return 8 * params.height**2

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        key, *keys = jax.random.split(key, num=7)

        obj = jax.random.choice(keys[0], _allowed_entities)
        door_color, obj_color = jax.random.choice(keys[1], _allowed_colors, shape=(2,))
        door_pos = jax.random.randint(keys[2], shape=(), minval=1, maxval=params.height - 1)

        grid = two_rooms(params.height, params.width)
        grid = grid.at[door_pos, params.width // 2].set(TILES_REGISTRY[Tiles.DOOR_LOCKED, door_color])

        # mask out positions after the wall, so that agent and key are always on the same side
        # WARN: this is a bit expensive, judging by the FPS benchmark
        mask = coordinates_mask(grid, (params.height, params.width // 2), comparison_fn=jnp.less)
        key_coords, agent_coords = sample_coordinates(keys[3], grid, num=2, mask=mask)

        mask = coordinates_mask(grid, (0, params.width // 2 + 1), comparison_fn=jnp.greater_equal)
        obj_coords = sample_coordinates(keys[4], grid, num=1, mask=mask).squeeze()

        grid = grid.at[key_coords[0], key_coords[1]].set(TILES_REGISTRY[Tiles.KEY, door_color])
        grid = grid.at[obj_coords[0], obj_coords[1]].set(TILES_REGISTRY[obj, obj_color])

        agent = AgentState(position=agent_coords, direction=sample_direction(keys[5]))
        goal_encoding = AgentHoldGoal(tile=TILES_REGISTRY[obj, obj_color]).encode()

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
