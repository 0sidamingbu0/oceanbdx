#!/usr/bin/env python3
"""Regression tests for the paper-aligned puppeteering mapping."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gamepad_input import GamepadSnapshot, PuppeteeringMapper  # noqa: E402


class PuppeteeringMapperTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "axis_deadzone": 0.08,
            "walk_button_behavior": "toggle",
            "r1_hold_duration_s": 0.35,
            "normal_walk_gain": 0.5,
            "full_walk_gain": 1.0,
            "posture_rearm_duration_s": 0.04,
            "gaze_torso_threshold": 0.75,
            "mapping": {
                "axis_left_x": 0,
                "axis_left_y": 1,
                "axis_right_x": 2,
                "axis_right_y": 3,
                "axis_dpad_x": 6,
                "axis_dpad_y": 7,
                "button_a": 0,
                "button_r1": 7,
                "button_l2": 8,
                "button_r2": 9,
                "button_start": 11,
            },
        }
        self.max_vel = np.array([0.25, 0.15, 0.8])
        self.torso_min = np.array([-0.04, -0.17, -0.24, -0.09])
        self.torso_max = np.array([0.01, 0.17, 0.24, 0.09])
        self.max_head = np.array([0.007, 0.17, 0.33, 0.20])
        self.mapper = PuppeteeringMapper(
            self.cfg, self.max_vel, self.torso_min, self.torso_max, self.max_head
        )
        self.mapper.update(self.snapshot(), 0.05)

    @staticmethod
    def snapshot(axes=None, buttons=None, connected=True):
        axis_values = np.zeros(8, dtype=np.float32)
        button_values = np.zeros(16, dtype=np.int8)
        if axes:
            for index, value in axes.items():
                axis_values[index] = value
        if buttons:
            for index, value in buttons.items():
                button_values[index] = value
        return GamepadSnapshot(axis_values, button_values, connected)

    def short_press_r1(self):
        self.mapper.update(self.snapshot(buttons={7: 1}), 0.05)
        return self.mapper.update(self.snapshot(), 0.01)

    def test_default_and_disconnect_are_standing_zero(self):
        command = self.mapper.update(self.snapshot(), 0.02)
        self.assertFalse(command.walk_requested)
        np.testing.assert_allclose(command.walk_command, 0.0)

        self.short_press_r1()
        command = self.mapper.update(GamepadSnapshot.zero(), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertFalse(command.connected)
        np.testing.assert_allclose(command.walk_command, 0.0)

    def test_short_r1_press_toggles_walk_and_uses_half_gain(self):
        command = self.short_press_r1()
        self.assertTrue(command.walk_requested)

        command = self.mapper.update(self.snapshot(axes={1: -1.0, 0: 1.0}), 0.02)
        self.assertAlmostEqual(command.walk_command[0], self.max_vel[0] * 0.5)
        self.assertAlmostEqual(command.walk_command[2], -self.max_vel[2] * 0.5)
        np.testing.assert_allclose(command.torso_command, 0.0)

        command = self.short_press_r1()
        self.assertFalse(command.walk_requested)

    def test_long_r1_hold_selects_full_gain_without_toggling(self):
        self.short_press_r1()
        command = None
        for _ in range(20):
            command = self.mapper.update(
                self.snapshot(axes={1: -1.0}, buttons={7: 1}), 0.02
            )
        self.assertIsNotNone(command)
        self.assertTrue(command.walk_requested)
        self.assertTrue(command.full_speed)
        self.assertAlmostEqual(command.walk_command[0], self.max_vel[0])

        command = self.mapper.update(self.snapshot(axes={1: -1.0}), 0.02)
        self.assertTrue(command.walk_requested)
        self.assertFalse(command.full_speed)
        self.assertAlmostEqual(command.walk_command[0], self.max_vel[0] * 0.5)

    def test_a_button_requests_standing(self):
        self.short_press_r1()
        command = self.mapper.update(self.snapshot(buttons={0: 1}), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertTrue(command.stand_requested)
        np.testing.assert_allclose(command.walk_command, 0.0)

    def test_a_consumes_an_overlapping_r1_gesture(self):
        self.short_press_r1()
        self.mapper.update(self.snapshot(buttons={7: 1}), 0.02)
        command = self.mapper.update(self.snapshot(buttons={0: 1}), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertTrue(command.stand_requested)

        command = self.mapper.update(self.snapshot(), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertFalse(command.stand_requested)

    def test_start_requests_standing_and_is_edge_triggered(self):
        self.short_press_r1()
        command = self.mapper.update(self.snapshot(buttons={11: 1}), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertTrue(command.stand_requested)
        self.assertTrue(command.start_requested)

        command = self.mapper.update(self.snapshot(buttons={11: 1}), 0.02)
        self.assertFalse(command.stand_requested)
        self.assertFalse(command.start_requested)

    def test_project_yaml_matches_the_tested_gamepad_indices(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config/oceanbdx.yaml"), encoding="utf-8") as stream:
            mapping = yaml.safe_load(stream)["oceanbdx"]["puppeteering"]["mapping"]
        self.assertEqual(
            mapping,
            {
                "axis_left_x": 0,
                "axis_left_y": 1,
                "axis_right_x": 2,
                "axis_right_y": 3,
                "axis_dpad_x": 6,
                "axis_dpad_y": 7,
                "button_a": 0,
                "button_r1": 7,
                "button_l2": 8,
                "button_r2": 9,
                "button_start": 11,
            },
        )

    def test_standing_left_stick_moves_torso_and_counter_rotates_head(self):
        command = self.mapper.update(self.snapshot(axes={1: -1.0}), 0.02)
        self.assertAlmostEqual(command.torso_command[1], self.torso_max[1])
        self.assertAlmostEqual(command.head_command[1], -self.torso_max[1])

        command = self.mapper.update(self.snapshot(axes={0: 1.0}), 0.02)
        self.assertAlmostEqual(command.torso_command[2], self.torso_min[2])
        self.assertAlmostEqual(command.head_command[2], -self.torso_min[2])

        command = self.mapper.update(self.snapshot(axes={1: 1.0}), 0.02)
        self.assertAlmostEqual(command.torso_command[0], self.torso_min[0])
        self.assertAlmostEqual(command.torso_command[1], 0.0)

    def test_right_stick_adds_torso_only_near_gaze_limit(self):
        command = self.mapper.update(self.snapshot(axes={2: 0.5}), 0.02)
        self.assertAlmostEqual(command.torso_command[2], 0.0)
        self.assertLess(command.head_command[2], 0.0)

        threshold_raw = self.cfg["axis_deadzone"] + self.cfg["gaze_torso_threshold"] * (
            1.0 - self.cfg["axis_deadzone"]
        )
        command = self.mapper.update(self.snapshot(axes={2: threshold_raw}), 0.02)
        self.assertAlmostEqual(command.torso_command[2], 0.0)
        self.assertAlmostEqual(command.head_command[2], -self.max_head[2])

        command = self.mapper.update(self.snapshot(axes={2: 1.0}), 0.02)
        self.assertAlmostEqual(command.torso_command[2], self.torso_min[2])
        self.assertAlmostEqual(command.head_command[2], -self.max_head[2])

    def test_triggers_map_to_lateral_walk_or_standing_roll(self):
        command = self.mapper.update(self.snapshot(buttons={8: 1}), 0.02)
        self.assertAlmostEqual(command.torso_command[3], self.torso_max[3])
        self.assertAlmostEqual(command.head_command[3], -self.torso_max[3])

        self.short_press_r1()
        command = self.mapper.update(self.snapshot(buttons={9: 1}), 0.02)
        self.assertAlmostEqual(command.walk_command[1], -self.max_vel[1] * 0.5)

    def test_hold_mode_walks_only_while_r1_is_pressed(self):
        cfg = dict(self.cfg)
        cfg["walk_button_behavior"] = "hold"
        mapper = PuppeteeringMapper(
            cfg, self.max_vel, self.torso_min, self.torso_max, self.max_head
        )
        command = mapper.update(self.snapshot(axes={1: -1.0}, buttons={7: 1}), 0.02)
        self.assertTrue(command.walk_requested)
        self.assertFalse(command.full_speed)
        self.assertAlmostEqual(command.walk_command[0], self.max_vel[0] * 0.5)

        command = mapper.update(self.snapshot(axes={1: -1.0}), 0.02)
        self.assertFalse(command.walk_requested)
        self.assertFalse(mapper.posture_inputs_armed)
        np.testing.assert_allclose(command.torso_command, 0.0)

        for _ in range(10):
            command = mapper.update(self.snapshot(axes={1: -1.0}), 0.02)
        self.assertFalse(mapper.posture_inputs_armed)
        np.testing.assert_allclose(command.torso_command, 0.0)

        for _ in range(3):
            mapper.update(self.snapshot(), 0.02)
        self.assertTrue(mapper.posture_inputs_armed)
        command = mapper.update(self.snapshot(axes={1: -1.0}), 0.02)
        self.assertAlmostEqual(command.torso_command[1], self.torso_max[1])

    def test_hold_mode_a_button_overrides_held_r1(self):
        cfg = dict(self.cfg)
        cfg["walk_button_behavior"] = "hold"
        mapper = PuppeteeringMapper(
            cfg, self.max_vel, self.torso_min, self.torso_max, self.max_head
        )
        command = mapper.update(
            self.snapshot(axes={1: -1.0}, buttons={0: 1, 7: 1}), 0.02
        )
        self.assertFalse(command.walk_requested)
        self.assertTrue(command.stand_requested)
        np.testing.assert_allclose(command.walk_command, 0.0)

        command = mapper.update(
            self.snapshot(axes={1: -1.0}, buttons={7: 1}), 0.02
        )
        self.assertFalse(command.walk_requested)

        mapper.update(self.snapshot(), 0.02)
        command = mapper.update(
            self.snapshot(axes={1: -1.0}, buttons={7: 1}), 0.02
        )
        self.assertTrue(command.walk_requested)

    def test_startup_requires_neutral_before_standing_posture_control(self):
        mapper = PuppeteeringMapper(
            self.cfg, self.max_vel, self.torso_min, self.torso_max, self.max_head
        )
        command = mapper.update(self.snapshot(axes={0: 1.0}, buttons={8: 1}), 0.20)
        self.assertFalse(mapper.posture_inputs_armed)
        np.testing.assert_allclose(command.torso_command, 0.0)

        for _ in range(3):
            mapper.update(self.snapshot(), 0.02)
        self.assertTrue(mapper.posture_inputs_armed)
        command = mapper.update(self.snapshot(axes={0: 1.0}, buttons={8: 1}), 0.02)
        self.assertAlmostEqual(command.torso_command[2], self.torso_min[2])
        self.assertAlmostEqual(command.torso_command[3], self.torso_max[3])

    def test_active_policy_controls_axis_semantics_during_switch(self):
        self.mapper.walk_requested = True
        walking_input = self.snapshot(axes={1: -1.0}, buttons={8: 1})
        command = self.mapper.update(walking_input, 0.02, active_walking=True)
        np.testing.assert_allclose(command.torso_command, 0.0)

        command = self.mapper.update(
            self.snapshot(axes={1: -1.0}, buttons={0: 1, 8: 1}),
            0.02,
            active_walking=True,
        )
        self.assertFalse(command.walk_requested)
        np.testing.assert_allclose(command.torso_command, 0.0)

        command = self.mapper.update(walking_input, 0.02, active_walking=False)
        self.assertFalse(self.mapper.posture_inputs_armed)
        np.testing.assert_allclose(command.torso_command, 0.0)

        for _ in range(3):
            self.mapper.update(self.snapshot(), 0.02, active_walking=False)
        command = self.mapper.update(walking_input, 0.02, active_walking=False)
        self.assertAlmostEqual(command.torso_command[1], self.torso_max[1])
        self.assertAlmostEqual(command.torso_command[3], self.torso_max[3])


if __name__ == "__main__":
    unittest.main()
