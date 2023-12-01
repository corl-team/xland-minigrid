import jax
import jax.numpy as jnp

from .constants import FREE_TO_PUT_DOWN, LOS_BLOCKING, PICKABLE, TILES_REGISTRY, WALKABLE, Colors, Tiles


def empty_world(height, width):
    grid = jnp.zeros((height, width, 2), dtype=jnp.uint8)
    grid = grid.at[:, :, 0:2].set(TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK])
    return grid


# wait, is this just a jnp.array_equal?
def equal(tile1, tile2):
    return jnp.all(jnp.equal(tile1, tile2))


def get_neighbouring_tiles(grid, y, x):
    # end_of_map = TILES_REGISTRY[Tiles.END_OF_MAP, Colors.END_OF_MAP]
    end_of_map = Tiles.END_OF_MAP

    up_tile = grid.at[y - 1, x].get(mode="fill", fill_value=end_of_map)
    right_tile = grid.at[y, x + 1].get(mode="fill", fill_value=end_of_map)
    down_tile = grid.at[y + 1, x].get(mode="fill", fill_value=end_of_map)
    left_tile = grid.at[y, x - 1].get(mode="fill", fill_value=end_of_map)
    # up_tile = jax.lax.select(y > 0, grid[y - 1, x], end_of_map)
    # right_tile = jax.lax.select(x < grid.shape[1] - 2, grid[y, x + 1], end_of_map)
    # down_tile = jax.lax.select(y < grid.shape[0] - 2, grid[y + 1, x], end_of_map)
    # left_tile = jax.lax.select(x > 0, grid[y, x - 1], end_of_map)
    return up_tile, right_tile, down_tile, left_tile


def horizontal_line(grid, x, y, length, tile):
    grid = grid.at[y, x : x + length].set(tile)
    return grid


def vertical_line(grid, x, y, length, tile):
    grid = grid.at[y : y + length, x].set(tile)
    return grid


def rectangle(grid, x, y, h, w, tile):
    grid = vertical_line(grid, x, y, h, tile)
    grid = vertical_line(grid, x + w - 1, y, h, tile)
    grid = horizontal_line(grid, x, y, w, tile)
    grid = horizontal_line(grid, x, y + h - 1, w, tile)
    return grid


def room(height, width):
    grid = empty_world(height, width)
    grid = rectangle(grid, 0, 0, height, width, tile=TILES_REGISTRY[Tiles.WALL, Colors.GREY])
    return grid


def two_rooms(height, width):
    wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]

    grid = empty_world(height, width)
    grid = rectangle(grid, 0, 0, height, width, tile=wall_tile)
    grid = vertical_line(grid, width // 2, 0, height, tile=wall_tile)
    return grid


def four_rooms(height, width):
    wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]

    grid = empty_world(height, width)
    grid = rectangle(grid, 0, 0, height, width, tile=wall_tile)
    grid = vertical_line(grid, width // 2, 0, height, tile=wall_tile)
    grid = horizontal_line(grid, 0, height // 2, width, tile=wall_tile)
    return grid


def nine_rooms(height, width):
    wall_tile = TILES_REGISTRY[Tiles.WALL, Colors.GREY]

    grid = empty_world(height, width)
    grid = rectangle(grid, 0, 0, height, width, tile=wall_tile)
    grid = vertical_line(grid, width // 3, 0, height, tile=wall_tile)
    grid = vertical_line(grid, 2 * (width // 3), 0, height, tile=wall_tile)
    grid = horizontal_line(grid, 0, height // 3, width, tile=wall_tile)
    grid = horizontal_line(grid, 0, 2 * (height // 3), width, tile=wall_tile)
    return grid


def check_walkable(grid, position):
    tile_id = grid[position[0], position[1], 0]
    is_walkable = jnp.isin(tile_id, WALKABLE, assume_unique=True)

    return is_walkable


def check_pickable(grid, position):
    tile_id = grid[position[0], position[1], 0]
    is_pickable = jnp.isin(tile_id, PICKABLE, assume_unique=True)
    return is_pickable


def check_can_put(grid, position):
    tile_id = grid[position[0], position[1], 0]
    can_put = jnp.isin(tile_id, FREE_TO_PUT_DOWN, assume_unique=True)

    return can_put


def check_see_behind(grid, position):
    tile_id = grid[position[0], position[1], 0]
    is_not_blocking = jnp.isin(tile_id, LOS_BLOCKING, assume_unique=True, invert=True)

    return is_not_blocking


def align_with_up(grid, direction):
    aligned_grid = jax.lax.switch(
        direction,
        (
            lambda: grid,
            lambda: jnp.rot90(grid, 1),
            lambda: jnp.rot90(grid, 2),
            lambda: jnp.rot90(grid, 3),
        ),
    )
    return aligned_grid


def grid_coords(grid):
    coords = jnp.mgrid[: grid.shape[0], : grid.shape[1]]
    coords = coords.transpose(1, 2, 0).reshape(-1, 2)
    return coords


def transparent_mask(grid):
    coords = grid_coords(grid)
    mask = jax.vmap(check_see_behind, in_axes=(None, 0))(grid, coords)
    mask = mask.reshape(grid.shape[0], grid.shape[1])
    return mask


def free_tiles_mask(grid):
    coords = grid_coords(grid)
    mask = jax.vmap(check_can_put, in_axes=(None, 0))(grid, coords)
    mask = mask.reshape(grid.shape[0], grid.shape[1])
    return mask


def coordinates_mask(grid, address, comparison_fn):
    positions = jnp.mgrid[: grid.shape[0], : grid.shape[1]]
    cond_1 = comparison_fn(positions[0], address[0])
    cond_2 = comparison_fn(positions[1], address[1])
    mask = jnp.logical_and(cond_1, cond_2)
    return mask


def sample_coordinates(key, grid, num, mask=None):
    if mask is None:
        mask = jnp.ones((grid.shape[0], grid.shape[1]), dtype=jnp.bool_)

    coords = jax.random.choice(
        key=key,
        shape=(num,),
        a=jnp.arange(grid.shape[0] * grid.shape[1]),
        replace=False,
        p=(mask & free_tiles_mask(grid)).flatten(),
    )
    coords = jnp.divmod(coords, grid.shape[1])
    coords = jnp.concatenate((coords[0].reshape(-1, 1), coords[1].reshape(-1, 1)), axis=-1)
    return coords


def sample_direction(key):
    return jax.random.randint(key, shape=(), minval=0, maxval=4)


def pad_along_axis(arr, pad_to, axis=0, fill_value=0):
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return jnp.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def cartesian_product_1d(a, b):
    return jnp.dstack(jnp.meshgrid(a, b)).reshape(-1, 2)
