import jax
import jax.numpy as jnp
from flax import struct
from jax.random import KeyArray

from ..core.constants import TILES_REGISTRY, Colors, Tiles
from ..core.goals import EmptyGoal
from ..core.grid import (
    equal,
    four_rooms,
    horizontal_line,
    nine_rooms,
    room,
    sample_coordinates,
    sample_direction,
    two_rooms,
    vertical_line,
)
from ..core.rules import EmptyRule
from ..environment import Environment, EnvParams
from ..types import AgentState, EnvCarry, RuleSet, State

_empty_ruleset = RuleSet(
    goal=EmptyGoal().encode(),
    rules=EmptyRule().encode()[None, ...],
    init_tiles=jnp.array(((TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY],))),
)

_empty_tile = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
_wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
# colors for doors between rooms
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


# helper functions to generate various maps, inspired by the common minigrid layouts
# TODO: all worlds should be square
def generate_room(key: KeyArray, height, width):
    grid = room(height, width)
    return key, grid


def generate_two_rooms(key: KeyArray, height, width):
    key, color_key, door_key = jax.random.split(key, num=3)

    color = jax.random.choice(color_key, _allowed_colors)
    door_pos = jax.random.randint(door_key, shape=(), minval=1, maxval=height - 1)

    grid = two_rooms(height, width)
    grid = grid.at[door_pos, width // 2].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, color])

    return key, grid


def generate_four_rooms(key: KeyArray, height, width):
    key, doors_key, colors_key = jax.random.split(key, num=3)

    doors_offsets = jax.random.randint(doors_key, shape=(4,), minval=1, maxval=height // 2)
    colors = jax.random.choice(colors_key, _allowed_colors, shape=(4,))

    grid = four_rooms(height, width)
    grid = grid.at[height // 2, doors_offsets[0]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[0]])
    grid = grid.at[height // 2, width // 2 + doors_offsets[1]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[1]])
    grid = grid.at[doors_offsets[2], width // 2].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[2]])
    grid = grid.at[height // 2 + doors_offsets[3], width // 2].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[3]])

    return key, grid


def generate_six_rooms(key: KeyArray, height, width):
    key, colors_key = jax.random.split(key)

    grid = room(height, width)
    grid = vertical_line(grid, width // 2 - 2, 0, height, _wall_tile)
    grid = vertical_line(grid, width // 2 + 2, 0, height, _wall_tile)

    for i in range(1, 3):
        grid = horizontal_line(grid, 0, i * (height // 3), width // 2 - 2, _wall_tile)
        grid = horizontal_line(grid, width // 2 + 2, i * (height // 3), width // 2 - 2, _wall_tile)

    doors_idxs = (
        # left doors
        (height // 2 - (height // 3), width // 2 - 2),
        (height // 2, width // 2 - 2),
        (height // 2 + (height // 3), width // 2 - 2),
        # right doors
        (height // 2 - (height // 3), width // 2 + 2),
        (height // 2, width // 2 + 2),
        (height // 2 + (height // 3), width // 2 + 2),
    )
    colors = jax.random.choice(colors_key, _allowed_colors, shape=(6,))

    for i in range(6):
        grid = grid.at[doors_idxs[i][0], doors_idxs[i][1]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[i]])

    return key, grid


def generate_nine_rooms(key: KeyArray, height, width):
    # valid sizes should follow 3 * x + 4: 7, 10, 13, 16, 19, 22, 25, 28, 31, ...
    # (size - 4) % 3 == 0
    key, doors_key, colors_key = jax.random.split(key, num=3)
    roomW, roomH = width // 3, height // 3

    grid = nine_rooms(height, width)
    # assuming that rooms are square!
    door_coords = jax.random.randint(doors_key, shape=(12,), minval=1, maxval=roomW)
    colors = jax.random.choice(colors_key, _allowed_colors, shape=(12,))

    # adapted from minigrid playground
    door_idx = 0
    for i in range(0, 3):
        for j in range(0, 3):
            xL = i * roomW
            yT = j * roomH
            xR = xL + roomW
            yB = yT + roomH

            if i + 1 < 3:
                grid = grid.at[yT + door_coords[door_idx], xR].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[door_idx]])
                door_idx = door_idx + 1

            if j + 1 < 3:
                grid = grid.at[yB, xL + door_coords[door_idx]].set(TILES_REGISTRY[Tiles.DOOR_CLOSED, colors[door_idx]])
                door_idx = door_idx + 1

    return key, grid


class XLandMiniGridEnvOptions(EnvParams):
    # you can vmap on rulesets for multi-task/meta learning
    ruleset: RuleSet = struct.field(pytree_node=True, default=_empty_ruleset)
    # experimental (can not vmap on it)
    grid_type: int = struct.field(pytree_node=False, default="1R")


class XLandMiniGrid(Environment):
    def default_params(self, **kwargs) -> EnvParams:
        default_params = XLandMiniGridEnvOptions(view_size=5)
        return default_params.replace(**kwargs)

    def time_limit(self, params: XLandMiniGridEnvOptions) -> int:
        # this is just a heuristic to prevent brute force in one episode,
        # agent need to remember what he tried in previous episodes.
        # If this is too small, change it or increase number of trials (these are not equivalent).
        return 3 * (params.height * params.width)

    def _generate_problem(self, params: XLandMiniGridEnvOptions, key: KeyArray) -> State:
        # WARN: we can make this compatible with jit (to vmap on different layouts during training),
        # but it will probably be very costly, as lax.switch will generate all layouts during reset under vmap
        # TODO: experiment with this under jit, does it possible to make it jit-compatible without overhead?
        if params.grid_type == "R1":
            key, grid = generate_room(key, params.height, params.width)
        elif params.grid_type == "R2":
            key, grid = generate_two_rooms(key, params.height, params.width)
        elif params.grid_type == "R4":
            key, grid = generate_four_rooms(key, params.height, params.width)
        elif params.grid_type == "R6":
            key, grid = generate_six_rooms(key, params.height, params.width)
        elif params.grid_type == "R9":
            key, grid = generate_nine_rooms(key, params.height, params.width)
        else:
            # WARN: will not work under jit!
            raise RuntimeError('Unknown grid type, should be one of: ["R1", "R2", "R4", "R6", "R9"]')

        num_objects = len(params.ruleset.init_tiles)
        objects = params.ruleset.init_tiles

        key, coords_key, dir_key = jax.random.split(key, num=3)
        positions = sample_coordinates(coords_key, grid, num=num_objects + 1)
        for i in range(num_objects):
            # we ignore empty tiles, as they are just paddings to the same shape
            grid = jax.lax.select(
                equal(objects[i], TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]),
                grid,
                grid.at[positions[i][0], positions[i][1]].set(objects[i]),
            )

        agent = AgentState(position=positions[-1], direction=sample_direction(dir_key))
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=params.ruleset.goal,
            rule_encoding=params.ruleset.rules,
            carry=EnvCarry(),
        )
        return state
