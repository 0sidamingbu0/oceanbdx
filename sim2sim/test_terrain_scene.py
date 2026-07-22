#!/usr/bin/env python3
"""Regression tests for the deterministic MuJoCo terrain test arena."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim2sim"))

from scripts import urdf2mjcf as terrain  # noqa: E402
import mujoco_sim as ms  # noqa: E402


class TerrainSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(
            os.path.join(ROOT, "sim2sim", "ocean_scene.xml")
        )
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)
        cls.terrain_geom_group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)

    def terrain_height(self, x, y):
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            self.model,
            self.data,
            np.array([x, y, 2.0]),
            np.array([0.0, 0.0, -1.0]),
            self.terrain_geom_group,
            1,
            -1,
            geom_id,
        )
        self.assertGreaterEqual(distance, 0.0)
        name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0])
        )
        return 2.0 - distance, name

    def test_scene_preserves_runtime_dynamics(self):
        expected_friction = np.array([0.5, 0.02, 0.001])
        for name in (
            "floor",
            "foot_r",
            "foot_l",
            "slope_terrain",
            "steep_slope_terrain",
            "rough_terrain",
        ):
            np.testing.assert_allclose(self.model.geom(name).friction, expected_friction)

        for name in terrain.LEG_JOINTS:
            dof = self.model.joint(name).dofadr[0]
            self.assertAlmostEqual(self.model.dof_armature[dof], 0.0043)
            self.assertAlmostEqual(self.model.dof_damping[dof], 0.0)
        for name in terrain.NECK_JOINTS:
            dof = self.model.joint(name).dofadr[0]
            self.assertAlmostEqual(self.model.dof_armature[dof], 0.01)
            self.assertAlmostEqual(self.model.dof_damping[dof], 0.05)

    def test_slope_has_flat_approach_and_five_degree_profile(self):
        height, name = self.terrain_height(terrain.SLOPE_X_MAX + 0.05, 0.0)
        self.assertEqual(name, "floor")
        self.assertAlmostEqual(height, 0.0, places=6)

        half_ramp_x = terrain.SLOPE_X_MAX - terrain.SLOPE_RAMP_LENGTH / 2.0
        height, name = self.terrain_height(half_ramp_x, 0.0)
        self.assertEqual(name, "slope_terrain")
        expected = (
            terrain.TERRAIN_Z_OFFSET
            + 0.5 * terrain.SLOPE_RAMP_LENGTH
            * np.tan(np.deg2rad(terrain.SLOPE_ANGLE_DEG))
        )
        self.assertAlmostEqual(height, expected, places=5)

        plateau_x = (terrain.SLOPE_X_MIN + terrain.SLOPE_X_MAX) / 2.0
        height, name = self.terrain_height(plateau_x, 0.0)
        self.assertEqual(name, "slope_terrain")
        self.assertAlmostEqual(
            height, terrain.TERRAIN_Z_OFFSET + terrain.SLOPE_PEAK_HEIGHT, places=5
        )

        half_descent_x = terrain.SLOPE_X_MIN + terrain.SLOPE_RAMP_LENGTH / 2.0
        height, name = self.terrain_height(half_descent_x, 0.0)
        self.assertEqual(name, "slope_terrain")
        self.assertAlmostEqual(height, expected, places=5)

    def test_rough_patch_matches_training_resolution_and_range(self):
        first = terrain.make_rough_heights()
        second = terrain.make_rough_heights()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (51, 51))
        self.assertAlmostEqual(float(first.min()), 0.0)
        self.assertAlmostEqual(float(first.max()), terrain.ROUGH_HEIGHT_MAX)
        np.testing.assert_allclose(first[0, :], 0.0)
        np.testing.assert_allclose(first[-1, :], 0.0)
        np.testing.assert_allclose(first[:, 0], 0.0)
        np.testing.assert_allclose(first[:, -1], 0.0)

        interior = first[
            terrain.ROUGH_BLEND_CELLS : -terrain.ROUGH_BLEND_CELLS,
            terrain.ROUGH_BLEND_CELLS : -terrain.ROUGH_BLEND_CELLS,
        ]
        quantized = interior / terrain.ROUGH_HEIGHT_STEP
        np.testing.assert_allclose(quantized, np.round(quantized), atol=1.0e-12)
        self.assertGreater(float(np.std(interior)), 0.005)

        center_height, name = self.terrain_height(*terrain.ROUGH_CENTER)
        self.assertEqual(name, "rough_terrain")
        self.assertGreaterEqual(center_height, terrain.TERRAIN_Z_OFFSET)
        self.assertLessEqual(
            center_height,
            terrain.TERRAIN_Z_OFFSET + terrain.ROUGH_HEIGHT_MAX + 1.0e-6,
        )

    def test_steep_slope_matches_configured_profile(self):
        approach_x = terrain.STEEP_SLOPE_X_MAX + 0.05
        height, name = self.terrain_height(approach_x, terrain.STEEP_SLOPE_CENTER_Y)
        self.assertEqual(name, "floor")
        self.assertAlmostEqual(height, 0.0, places=6)

        half_ramp_x = (
            terrain.STEEP_SLOPE_X_MAX - terrain.STEEP_SLOPE_RAMP_LENGTH / 2.0
        )
        height, name = self.terrain_height(
            half_ramp_x, terrain.STEEP_SLOPE_CENTER_Y
        )
        self.assertEqual(name, "steep_slope_terrain")
        expected = (
            terrain.TERRAIN_Z_OFFSET
            + 0.5
            * terrain.STEEP_SLOPE_RAMP_LENGTH
            * np.tan(np.deg2rad(terrain.STEEP_SLOPE_ANGLE_DEG))
        )
        self.assertAlmostEqual(height, expected, places=5)

        plateau_x = (
            terrain.STEEP_SLOPE_X_MIN + terrain.STEEP_SLOPE_X_MAX
        ) / 2.0
        height, name = self.terrain_height(plateau_x, terrain.STEEP_SLOPE_CENTER_Y)
        self.assertEqual(name, "steep_slope_terrain")
        self.assertAlmostEqual(
            height,
            terrain.TERRAIN_Z_OFFSET + terrain.STEEP_SLOPE_PEAK_HEIGHT,
            places=5,
        )

    def test_spawn_and_routes_are_separate(self):
        spawn_height, name = self.terrain_height(0.4, 0.4)
        self.assertEqual(name, "floor")
        self.assertAlmostEqual(spawn_height, 0.0, places=6)

        slope_y_min = -terrain.SLOPE_WIDTH / 2.0
        rough_y_max = terrain.ROUGH_CENTER[1] + terrain.ROUGH_SIZE / 2.0
        self.assertLess(rough_y_max, slope_y_min)
        slope_y_max = terrain.SLOPE_WIDTH / 2.0
        steep_y_min = terrain.STEEP_SLOPE_CENTER_Y - terrain.STEEP_SLOPE_WIDTH / 2.0
        self.assertLess(slope_y_max, steep_y_min)

    def test_contact_diagnostics_accept_every_ground_region(self):
        args = SimpleNamespace(
            config=os.path.join(ROOT, "config", "oceanbdx.yaml"),
            policy=None,
            stand_policy=None,
            no_policy=True,
            real=False,
            manual=False,
            no_log=True,
            sim_rl_kd=None,
            sim_rl_kd_list=None,
            sim_action_scale=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            sim = ms.Sim(args)
        expected = {sim.model.geom(name).id for name in ms.GROUND_GEOM_NAMES}
        self.assertSetEqual(set(sim.ground_geom_ids), expected)

        right_gid, left_gid = sim.foot_geom_ids
        ground_ids = {
            name: sim.model.geom(name).id for name in ms.GROUND_GEOM_NAMES
        }
        contacts = [
            SimpleNamespace(geom1=left_gid, geom2=ground_ids["slope_terrain"]),
            SimpleNamespace(geom1=ground_ids["rough_terrain"], geom2=right_gid),
            SimpleNamespace(
                geom1=left_gid, geom2=ground_ids["steep_slope_terrain"]
            ),
            SimpleNamespace(geom1=left_gid, geom2=sim.model.geom("stool").id),
            SimpleNamespace(geom1=ground_ids["floor"], geom2=right_gid),
        ]
        sim.data = SimpleNamespace(ncon=len(contacts), contact=contacts)
        forces = [5.0, 7.0, 11.0, 100.0, 3.0]

        def contact_force(_model, _data, index, output):
            output[:] = 0.0
            output[0] = forces[index]

        with mock.patch.object(mujoco, "mj_contactForce", side_effect=contact_force):
            left, right = sim.foot_contact_forces()
        self.assertEqual(left, 16.0)
        self.assertEqual(right, 10.0)


if __name__ == "__main__":
    unittest.main()
