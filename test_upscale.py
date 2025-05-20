import unittest

import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose, assert_array_equal

from upscale import get_box_score, upscale_and_expand_flattened_coords


class TestBox(unittest.TestCase):
    def test_inside_box(self):
        coordinates = jnp.array(
            [[1, 1], [2, 2], [3, 3], [-1, 0], [0, -1], [1, 2], [2, 1], [0, 1], [1, 0]]
        )
        box = jnp.array([[0, 0], [2, 2]])
        result = get_box_score(coordinates, box)
        expected = jnp.array(
            [True, False, False, False, False, False, False, True, True]
        )
        assert_array_equal(result, expected)

    def test_inside_box_float(self):
        coordinates = jnp.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        box = jnp.array([[0, 0], [2, 2]])
        result = get_box_score(coordinates, box)
        expected = jnp.array([True, False, False, False, False])
        assert_array_equal(result, expected)

    def test_inside_box_broadcast(self):
        coordinates = jnp.array(
            [
                [[1, 1], [2, 2], [3, 3]],
                [[-1, 0], [0, -1], [1, 2]],
                [[2, 1], [0, 1], [1, 0]],
            ]
        )
        box = jnp.array([[0, 0], [2, 2]])
        result = get_box_score(coordinates, box)
        expected = jnp.array(
            [
                [True, False, False],
                [False, False, False],
                [False, True, True],
            ]
        )
        assert_array_equal(result, expected)


class TestUpscaleExpand(unittest.TestCase):
    def test_single_pixel_no_flow_keep_all(self):
        coarse_f1 = self._grid_coords(1, 1)
        coarse_flow = self._same_flow(0.0, 0.0, 1)
        coarse_f2 = self._grid_coords(1, 1)
        confidence = self._same_confidence(0.5, 1)
        focus_box = self._entire_box(1, 1)
        image_size = self._size(1, 1)
        select_top = None
        f1, f2, priors, kept_f1, kept_f2 = upscale_and_expand_flattened_coords(
            coarse_f1,
            coarse_flow,
            coarse_f2,
            confidence,
            focus_box,
            image_size,
            select_top,
        )
        self._assert_maps_to(
            kept_f1, f1, y=0, x=0, maps_to=((0, 0), (1, 0), (0, 1), (1, 1))
        )
        self._assert_maps_to(
            kept_f2, f2, y=0, x=0, maps_to=((0, 0), (1, 0), (0, 1), (1, 1))
        )

    def test_single_pixel_flow_clip_keep_all(self):
        coarse_f1 = self._grid_coords(1, 1)
        coarse_flow = self._same_flow(0.6, 0.2, 1)
        coarse_f2 = self._grid_coords(1, 1)
        confidence = self._same_confidence(0.5, 1)
        focus_box = self._entire_box(2, 2)
        image_size = self._size(1, 1)
        select_top = None
        f1, f2, priors, kept_f1, kept_f2 = upscale_and_expand_flattened_coords(
            coarse_f1,
            coarse_flow,
            coarse_f2,
            confidence,
            focus_box,
            image_size,
            select_top,
        )
        self._assert_maps_to(
            kept_f1, f1, y=0, x=0, maps_to=((0, 0), (1, 0), (0, 1), (1, 1))
        )
        self._assert_maps_to(
            kept_f2, f2, y=0, x=0, maps_to=((1, 0), (1, 0), (1, 1), (1, 1))
        )

    def test_flow_clip_drop_focus(self):
        coarse_f1 = self._grid_coords(2, 2)
        coarse_flow = self._same_flow(0.6, 0.6, 4)
        coarse_f2 = self._grid_coords(2, 2)
        confidence = self._same_confidence(0.5, 4)
        focus_box = self._entire_box(3, 4)  # top half
        image_size = self._size(2, 2)
        select_top = 2
        f1, f2, priors, kept_coarse_f1, kept_coarse_f2 = (
            upscale_and_expand_flattened_coords(
                coarse_f1,
                coarse_flow,
                coarse_f2,
                confidence,
                focus_box,
                image_size,
                select_top,
            )
        )

        self._assert_maps_to(
            kept_coarse_f2, f2, y=0, x=0, maps_to=((1, 1), (2, 1), (1, 2), (2, 2))
        )
        self._assert_maps_to(
            kept_coarse_f2, f2, y=0, x=1, maps_to=((1, 3), (2, 3), (1, 3), (2, 3))
        )
        self._assert_dropped(coarse_f1, kept_coarse_f1, ((1, 0), (1, 1)))

    def test_batch_keep(self):
        coarse_f1 = self._grid_coords(2, 2, batches=2)
        flow_1 = jnp.array([[[-0.6, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
        flow_2 = self._same_flow(0, 0, 4)
        coarse_flow = jnp.concatenate([flow_1, flow_2], axis=0)
        coarse_f2 = self._grid_coords(2, 2, batches=2)
        confidence = jnp.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.1]])
        focus_box = self._entire_box(3, 4)  # top half
        image_size = self._size(2, 2)
        select_top = 3
        f1, f2, priors, kept_coarse_f1, kept_coarse_f2 = (
            upscale_and_expand_flattened_coords(
                coarse_f1,
                coarse_flow,
                coarse_f2,
                confidence,
                focus_box,
                image_size,
                select_top,
            )
        )
        self._assert_maps_to(
            kept_coarse_f2,
            f2,
            y=1,
            x=1,
            maps_to=((2, 2), (3, 2), (2, 3), (3, 3)),
            batch=0,
        )
        self._assert_maps_to(
            kept_coarse_f2,
            f2,
            y=0,
            x=0,
            maps_to=((0, 0), (1, 0), (0, 1), (1, 1)),
            batch=1,
        )
        self._assert_dropped(coarse_f1, kept_coarse_f1, ((0, 0),), batch=0)
        self._assert_dropped(coarse_f1, kept_coarse_f1, ((1, 1),), batch=1)

    def _same_flow(self, uy: float, ux: float, length: int, batches=1) -> jax.Array:
        single_flow = jnp.expand_dims(jnp.array([uy, ux], dtype=jnp.float32), (0, 1))
        flow_values = jnp.repeat(single_flow, length, axis=1)
        batched = jnp.repeat(flow_values, batches, axis=0)
        return batched

    def _grid_coords(self, height: int, width: int, batches=1) -> jax.Array:
        coords = jnp.stack(
            jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij"),
            axis=-1,
        ).reshape(-1, 2)
        single_batch = jnp.expand_dims(coords, 0)
        return jnp.repeat(single_batch, batches, axis=0)

    def _assert_maps_to(
        self,
        coarse: jax.Array,
        fine: jax.Array,
        *,
        y: int,
        x: int,
        maps_to: tuple,
        batch=0,
    ):
        source = [tuple(e.tolist()) for e in coarse[batch]]
        try:
            coarse_index = source.index((y, x))
        except ValueError:
            self.fail(f"{y, x} not found in {coarse.tolist()}")

        fine_index = coarse_index * 4
        existing_values = fine[batch, fine_index : fine_index + 4, :]
        expected_values = jnp.array(maps_to)
        assert_allclose(existing_values, expected_values)

    def _assert_dropped(self, all_coords, kept_coords, expected_dropped, batch=0):
        all_coords_set = set(tuple(e.tolist()) for e in all_coords[batch])
        kept_coords_set = set(tuple(e.tolist()) for e in kept_coords[batch])
        expected_dropped_set = set(expected_dropped)
        dropped = all_coords_set - kept_coords_set
        self.assertEqual(dropped, expected_dropped_set)

    def _same_confidence(self, confidence: float, length: int, batches=1) -> jax.Array:
        unbatched = jnp.array([confidence] * length)
        single_batch = jnp.expand_dims(unbatched, 0)
        return jnp.repeat(single_batch, batches, axis=0)

    def _size(self, height, width) -> jax.Array:
        return jnp.array([width, height], dtype=jnp.int32)

    def _entire_box(self, height: int, width: int):
        return jnp.array([[0, 0], [height, width]], dtype=jnp.int32)
