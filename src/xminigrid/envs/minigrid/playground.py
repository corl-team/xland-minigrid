import jax
import jax.numpy as jnp
from flax import struct

from ...core.constants import Colors, Tiles
from ...core.goals import EmptyGoal
from ...core.grid import cartesian_product_1d, nine_rooms, sample_coordinates, sample_direction
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

_goal_encoding = EmptyGoal().encode()
_rule_encoding = EmptyRule().encode()[None, ...]

# colors like in the original minigrid
_allowed_colors = jnp.array(
    (
        Colors.RED,
        Colors.GREEN,
        Colors.BLUE,
        Colors.PURPLE,
        Colors.YELLOW,
        Colors.GREY,
    ),
    dtype=jnp.uint8,
)

_allowed_doors = cartesian_product_1d(
    jnp.array((Tiles.DOOR_CLOSED,), dtype=jnp.uint8),
    _allowed_colors,
)
_allowed_objects = cartesian_product_1d(
    jnp.array((Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.KEY, Tiles.STAR, Tiles.HEX, Tiles.GOAL), dtype=jnp.uint8),
    _allowed_colors,
)
# number of doors with 9 rooms
_total_doors = 12


class PlaygroundEnvParams(EnvParams):
    num_objects: int = struct.field(pytree_node=False, default=12)


class Playground(Environment):
    def default_params(self, **kwargs) -> PlaygroundEnvParams:
        return PlaygroundEnvParams(height=19, width=19).replace(**kwargs)

    def time_limit(self, params: EnvParams) -> int:
        return 512

    def _generate_problem(self, params: PlaygroundEnvParams, key: jax.Array) -> State:
        key, *keys = jax.random.split(key, num=6)

        grid = nine_rooms(params.height, params.width)
        roomW, roomH = params.width // 3, params.height // 3

        # assuming that rooms are square!
        door_coords = jax.random.randint(keys[0], shape=(_total_doors,), minval=1, maxval=roomW)
        doors = jax.random.choice(keys[1], _allowed_doors, shape=(_total_doors,))

        # adapted from minigrid playground
        door_idx = 0
        for i in range(0, 3):
            for j in range(0, 3):
                xL = i * roomW
                yT = j * roomH
                xR = xL + roomW
                yB = yT + roomH

                if i + 1 < 3:
                    grid = grid.at[yT + door_coords[door_idx], xR].set(doors[door_idx])
                    door_idx = door_idx + 1

                if j + 1 < 3:
                    grid = grid.at[yB, xL + door_coords[door_idx]].set(doors[door_idx])
                    door_idx = door_idx + 1

        objects_coords = sample_coordinates(keys[2], grid, num=params.num_objects + 1)
        objects = jax.random.choice(keys[3], _allowed_objects, shape=(params.num_objects,))
        for i in range(params.num_objects):
            grid = grid.at[objects_coords[i][0], objects_coords[i][1]].set(objects[i])

        agent = AgentState(position=objects_coords[-1], direction=sample_direction(keys[4]))

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
