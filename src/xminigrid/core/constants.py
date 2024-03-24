import jax.numpy as jnp
from flax import struct

NUM_ACTIONS = 6

# GRID: [tile, color]
NUM_LAYERS = 2
NUM_TILES = 13
NUM_COLORS = 12


# enums, kinda...
class Tiles(struct.PyTreeNode):
    EMPTY: int = struct.field(pytree_node=False, default=0)
    FLOOR: int = struct.field(pytree_node=False, default=1)
    WALL: int = struct.field(pytree_node=False, default=2)
    BALL: int = struct.field(pytree_node=False, default=3)
    SQUARE: int = struct.field(pytree_node=False, default=4)
    PYRAMID: int = struct.field(pytree_node=False, default=5)
    GOAL: int = struct.field(pytree_node=False, default=6)
    KEY: int = struct.field(pytree_node=False, default=7)
    DOOR_LOCKED: int = struct.field(pytree_node=False, default=8)
    DOOR_CLOSED: int = struct.field(pytree_node=False, default=9)
    DOOR_OPEN: int = struct.field(pytree_node=False, default=10)
    HEX: int = struct.field(pytree_node=False, default=11)
    STAR: int = struct.field(pytree_node=False, default=12)


class Colors(struct.PyTreeNode):
    EMPTY: int = struct.field(pytree_node=False, default=0)
    RED: int = struct.field(pytree_node=False, default=1)
    GREEN: int = struct.field(pytree_node=False, default=2)
    BLUE: int = struct.field(pytree_node=False, default=3)
    PURPLE: int = struct.field(pytree_node=False, default=4)
    YELLOW: int = struct.field(pytree_node=False, default=5)
    GREY: int = struct.field(pytree_node=False, default=6)
    BLACK: int = struct.field(pytree_node=False, default=7)
    ORANGE: int = struct.field(pytree_node=False, default=8)
    WHITE: int = struct.field(pytree_node=False, default=9)
    BROWN: int = struct.field(pytree_node=False, default=10)
    PINK: int = struct.field(pytree_node=False, default=11)


# Only ~100 combinations so far, better to preallocate them
TILES_REGISTRY = []
for tile_id in range(NUM_TILES):
    for color_id in range(NUM_COLORS):
        TILES_REGISTRY.append((tile_id, color_id))
TILES_REGISTRY = jnp.array(TILES_REGISTRY, dtype=jnp.uint8).reshape(NUM_TILES, NUM_COLORS, -1)


DIRECTIONS = jnp.array(
    (
        (-1, 0),  # Up
        (0, 1),  # Right
        (1, 0),  # Down
        (0, -1),  # Left
    )
)


WALKABLE = jnp.array(
    (
        Tiles.FLOOR,
        Tiles.GOAL,
        Tiles.DOOR_OPEN,
    )
)

PICKABLE = jnp.array(
    (
        Tiles.BALL,
        Tiles.SQUARE,
        Tiles.PYRAMID,
        Tiles.KEY,
        Tiles.HEX,
        Tiles.STAR,
    )
)

FREE_TO_PUT_DOWN = jnp.array((Tiles.FLOOR,))

LOS_BLOCKING = jnp.array(
    (
        Tiles.WALL,
        Tiles.DOOR_LOCKED,
        Tiles.DOOR_CLOSED,
    )
)
