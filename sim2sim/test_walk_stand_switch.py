#!/usr/bin/env python3
"""Regression tests for real-signal-compatible stand/walk transitions."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco_sim as ms  # noqa: E402


class WalkStandSwitchTest(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(
            config=os.path.join(ms.ROOT, "config/oceanbdx.yaml"),
            policy=None,
            stand_policy=None,
            no_policy=False,
            real=False,
            manual=False,
            no_log=True,
            sim_rl_kd=None,
            sim_rl_kd_list=None,
            sim_action_scale=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim = ms.Sim(args)
        self.sim.state = "RL_WALK"
        self.sim.policy = self.sim.walk_policy
        self.sim.cmd[:] = [0.20, 0.0, 0.0]
        self.sim._effective_walk_cmd[:] = self.sim.cmd
        self.sim.rl_target = self.sim.stand_pose.copy()
        self.sim.rl_tick = 0
        self.q = self.sim.stand_pose.copy()
        self.dq = np.zeros(self.sim.nj)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.gyro = np.zeros(3)

    def update(self, count=1):
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(count):
                self.sim._update_pending_rl_switch(
                    self.q, self.dq, self.quat, self.gyro
                )

    def request(self, target):
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._switch_rl_state(target)

    def gamepad_update(self, snapshot, count=1):
        self.sim.gamepad_enabled = True
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(count):
                self.sim._apply_puppeteering_snapshot(snapshot)
                self.sim._update_puppeteering_switch(
                    self.q, self.dq, self.quat, self.gyro
                )
                self.sim._update_pending_rl_switch(
                    self.q, self.dq, self.quat, self.gyro
                )

    @staticmethod
    def gamepad_snapshot(axes=None, buttons=None, connected=True):
        axis_values = np.zeros(8, dtype=np.float32)
        button_values = np.zeros(16, dtype=np.int8)
        if axes:
            for index, value in axes.items():
                axis_values[index] = value
        if buttons:
            for index, value in buttons.items():
                button_values[index] = value
        return ms.GamepadSnapshot(axis_values, button_values, connected)

    def set_stand(self, torso_command=None):
        self.sim._clear_walk_stop_transition()
        self.sim._clear_stand_to_walk_transition()
        self.sim.state = "RL_STAND"
        self.sim.policy = self.sim.stand_policy
        command = np.zeros(4, dtype=np.float32)
        if torso_command is not None:
            command[:] = torso_command
        self.sim.torso_cmd[:] = command
        self.sim._effective_torso_cmd[:] = command
        self.sim.cmd[:] = 0.0
        self.sim._effective_walk_cmd[:] = 0.0
        self.sim.rl_target = self.sim.stand_pose.copy()
        self.sim.rl_tick = 0

    def test_phase_windows_are_inside_reference_double_support(self):
        self.sim.walk_policy.gait_phase = 0.05
        self.assertTrue(self.sim._phase_in_switch_window())
        self.sim.walk_policy.gait_phase = 0.55
        self.assertTrue(self.sim._phase_in_switch_window())
        self.sim.walk_policy.gait_phase = 0.25
        self.assertFalse(self.sim._phase_in_switch_window())
        self.sim.walk_policy.gait_phase = 0.75
        self.assertFalse(self.sim._phase_in_switch_window())

    def test_deceleration_is_monotonic_and_keeps_phase_moving(self):
        self.request("RL_STAND")
        peaks = []
        while self.sim._walk_stop_stage != "WAIT_PHASE":
            self.update()
            peaks.append(float(np.max(np.abs(self.sim._effective_walk_cmd))))
        self.assertTrue(np.all(np.diff(peaks) <= 1.0e-12))
        self.assertGreaterEqual(peaks[-1], self.sim.switch_min_moving_command - 1.0e-12)
        self.assertGreater(peaks[-1], self.sim.move_command_threshold)

    def test_transition_uses_no_contact_force_or_true_base_velocity(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("MuJoCo-only truth was read by the switch FSM")

        self.sim.foot_contact_forces = forbidden
        self.sim.base_linear_velocities = forbidden
        self.request("RL_STAND")
        for _ in range(1000):
            if self.sim._walk_stop_stage == "WAIT_PHASE":
                self.sim.walk_policy.gait_phase = 0.05
            self.update()
            if self.sim.state == "RL_STAND":
                break
        self.assertEqual(self.sim.state, "RL_STAND")

    def test_stability_is_fail_closed_and_each_limit_can_block(self):
        stable, _ = self.sim._walk_to_stand_stable(
            self.q, self.dq, self.quat, self.gyro
        )
        self.assertTrue(stable)

        cases = [
            (self.q, self.dq, np.array([np.nan, 0.0, 0.0, 0.0]), self.gyro),
            (self.q, self.dq, self.quat, np.array([self.sim.switch_gyro_xy_max, 0.0, 0.0])),
            (self.q, self.dq, self.quat, np.array([0.0, 0.0, self.sim.switch_gyro_z_max])),
            (
                self.q,
                np.full(self.sim.nj, self.sim.switch_joint_vel_rms_max),
                self.quat,
                self.gyro,
            ),
            (
                self.q + self.sim.switch_joint_pos_error_max,
                self.dq,
                self.quat,
                self.gyro,
            ),
        ]
        for q, dq, quat, gyro in cases:
            with self.subTest(q=q, dq=dq, quat=quat, gyro=gyro):
                allowed, _ = self.sim._walk_to_stand_stable(q, dq, quat, gyro)
                self.assertFalse(allowed)

        # Low-kP policies intentionally use target-q offset to generate support torque.
        # It is controller state, not a physical stability signal, and must not block switching.
        original_target = self.sim.rl_target.copy()
        self.sim.rl_target = self.q + 0.8
        allowed, _ = self.sim._walk_to_stand_stable(
            self.q, self.dq, self.quat, self.gyro
        )
        self.assertTrue(allowed)
        self.sim.rl_target[0] = np.nan
        allowed, _ = self.sim._walk_to_stand_stable(
            self.q, self.dq, self.quat, self.gyro
        )
        self.sim.rl_target = original_target
        self.assertFalse(allowed)

    def test_stability_confirmation_must_be_continuous(self):
        self.sim.pending_rl_state = "RL_STAND"
        self.sim._walk_stop_stage = "ZERO_HOLD"
        self.sim.switch_zero_hold_duration = 0.0
        self.sim.switch_stable_confirm_duration = 4 * self.sim.control_dt
        self.update(3)
        self.assertAlmostEqual(
            self.sim._walk_stop_stable_elapsed, 3 * self.sim.control_dt
        )
        self.gyro[0] = self.sim.switch_gyro_xy_max
        self.update()
        self.assertEqual(self.sim._walk_stop_stable_elapsed, 0.0)
        self.assertEqual(self.sim.state, "RL_WALK")

    def test_operator_cancel_resumes_command_smoothly(self):
        self.request("RL_STAND")
        self.update(20)
        stopped_at = self.sim._effective_walk_cmd.copy()
        self.request("RL_WALK")
        self.update()
        self.assertIsNone(self.sim.pending_rl_state)
        self.assertLessEqual(
            np.max(
                np.abs(self.sim._effective_walk_cmd - stopped_at)
                - self.sim.walk_command_accel_limits * self.sim.control_dt
            ),
            1.0e-12,
        )
        self.sim.walk_policy.gait_phase = 0.05
        self.update(
            int(
                np.ceil(
                    np.max(
                        np.abs(self.sim.cmd - self.sim._effective_walk_cmd)
                        / self.sim.walk_command_accel_limits
                    )
                    / self.sim.control_dt
                )
            )
            + 2
        )
        np.testing.assert_allclose(self.sim._effective_walk_cmd, self.sim.cmd)

    def test_keyboard_only_changes_target_command(self):
        self.sim.cmd[:] = 0.0
        self.sim._effective_walk_cmd[:] = 0.0
        phase_before = self.sim.walk_policy.gait_phase
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._cmd_key("w")
        self.assertAlmostEqual(self.sim.cmd[0], 0.1)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)
        self.assertEqual(self.sim.walk_policy.gait_phase, phase_before)
        self.update()
        self.assertGreater(self.sim._effective_walk_cmd[0], 0.0)
        self.assertLessEqual(
            self.sim._effective_walk_cmd[0],
            self.sim.walk_command_accel_limits[0] * self.sim.control_dt + 1.0e-12,
        )

    def test_viewer_push_keys_apply_both_directions_and_preserve_mouse_force(self):
        self.set_stand()
        self.sim.viewer_push_force[:] = [12.0, -34.0, 5.0]
        self.sim.viewer_push_duration = 0.1
        body_id = self.sim.viewer_push_base_body_id
        baseline = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6])
        self.sim.data.xfrc_applied[body_id] = baseline

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._cmd_key("5")
            self.sim._consume_viewer_push_requests()
        self.sim.data.qfrc_applied[:] = 0.0
        applied = self.sim._begin_viewer_push_substep()
        self.sim._apply_viewer_push_substep(applied)
        np.testing.assert_allclose(
            self.sim.data.qfrc_applied[:3], self.sim.viewer_push_force
        )
        self.sim._end_viewer_push_substep(applied)
        np.testing.assert_allclose(self.sim.data.xfrc_applied[body_id], baseline)

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._cmd_key("5")
            self.sim._cmd_key("6")
            self.sim._consume_viewer_push_requests()
        self.sim.data.qfrc_applied[:] = 0.0
        applied = self.sim._begin_viewer_push_substep()
        self.sim._apply_viewer_push_substep(applied)
        np.testing.assert_allclose(
            self.sim.data.qfrc_applied[:3], -self.sim.viewer_push_force
        )
        self.sim._end_viewer_push_substep(applied)
        np.testing.assert_allclose(self.sim.data.xfrc_applied[body_id], baseline)

    def test_viewer_push_duration_and_safety_guards(self):
        self.set_stand()
        self.sim.viewer_push_force[:] = [0.0, 60.0, 0.0]
        timestep = float(self.sim.model.opt.timestep)
        self.sim.viewer_push_duration = 2.5 * timestep

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertTrue(self.sim._trigger_viewer_push())
            substeps = 0
            while True:
                applied = self.sim._begin_viewer_push_substep()
                if applied is None:
                    break
                self.sim._end_viewer_push_substep(applied)
                substeps += 1
        self.assertEqual(substeps, 3)
        self.assertEqual(output.getvalue().count("[viewer_push] complete"), 1)
        self.assertEqual(self.sim._viewer_push_remaining, 0.0)

        self.sim.state = "SIT"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.sim._trigger_viewer_push())
        self.set_stand()
        self.sim.want_real = True
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.sim._trigger_viewer_push())
        self.assertEqual(self.sim._viewer_push_remaining, 0.0)

    def test_viewer_push_configuration_validation_and_reset(self):
        base_args = dict(
            config=os.path.join(ms.ROOT, "config/oceanbdx.yaml"),
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
        args = SimpleNamespace(
            **base_args,
            viewer_push_force_x=11.0,
            viewer_push_force_y=-22.0,
            viewer_push_force_z=3.0,
            viewer_push_duration=0.125,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            sim = ms.Sim(args)
        np.testing.assert_allclose(sim.viewer_push_force, [11.0, -22.0, 3.0])
        self.assertEqual(sim.viewer_push_duration, 0.125)
        self.assertEqual(sim.viewer_push_base_body_id, sim.model.body("base_link").id)

        sim.state = "RL_STAND"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(sim._trigger_viewer_push())
            sim._request_viewer_push(-1.0)
            sim.reset()
            sim._consume_viewer_push_requests()
        self.assertEqual(sim._viewer_push_remaining, 0.0)

        invalid_cases = (
            {"viewer_push_force_x": float("nan")},
            {"viewer_push_force_z": float("inf")},
            {"viewer_push_duration": 0.0},
            {"viewer_push_duration": -0.1},
            {"viewer_push_duration": float("nan")},
        )
        for override in invalid_cases:
            invalid_args = dict(base_args)
            invalid_args.update(
                viewer_push_force_x=0.0,
                viewer_push_force_y=0.0,
                viewer_push_force_z=0.0,
                viewer_push_duration=0.1,
            )
            invalid_args.update(override)
            with self.subTest(override=override):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        ms.Sim(SimpleNamespace(**invalid_args))

    def test_persistent_head_command_is_clipped_for_active_policy(self):
        self.sim.head_cmd[:] = self.sim.stand_max_head
        self.sim.state = "RL_WALK"
        np.testing.assert_allclose(
            self.sim._active_policy_head_command(), self.sim.walk_max_head
        )
        # Keep the requested standing pose intact so it can become active again after switching
        # back; only the observation sent to walking is clipped.
        np.testing.assert_allclose(self.sim.head_cmd, self.sim.stand_max_head)

        self.sim.state = "RL_STAND"
        np.testing.assert_allclose(
            self.sim._active_policy_head_command(), self.sim.stand_max_head
        )

    def test_max_command_stops_smoothly_at_double_support(self):
        self.sim.cmd[:] = 0.0
        self.sim._effective_walk_cmd[:] = [self.sim.max_vel[0], 0.0, 0.0]
        self.sim.walk_policy.gait_phase = 0.25
        previous = self.sim._effective_walk_cmd.copy()
        for _ in range(500):
            self.update()
            current = self.sim._effective_walk_cmd.copy()
            self.assertLessEqual(
                np.max(
                    np.abs(current - previous)
                    - self.sim.walk_command_decel_limits * self.sim.control_dt
                ),
                1.0e-12,
            )
            previous = current
            if np.max(np.abs(current)) <= self.sim.switch_min_moving_command + 1.0e-12:
                break
        self.assertGreater(
            np.max(np.abs(self.sim._effective_walk_cmd)),
            self.sim.move_command_threshold,
        )
        self.assertFalse(self.sim._phase_in_switch_window())

        self.sim.walk_policy.gait_phase = 0.05
        previous = self.sim._effective_walk_cmd.copy()
        for _ in range(500):
            self.update()
            current = self.sim._effective_walk_cmd.copy()
            self.assertLessEqual(
                np.max(
                    np.abs(current - previous)
                    - self.sim.walk_command_decel_limits * self.sim.control_dt
                ),
                1.0e-12,
            )
            previous = current
            if np.max(np.abs(self.sim._effective_walk_cmd)) <= 1.0e-12:
                break
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0, atol=1.0e-12)

    def test_direction_reversal_crosses_zero_only_at_double_support(self):
        self.sim.cmd[:] = [-self.sim.max_vel[0], 0.0, 0.0]
        self.sim._effective_walk_cmd[:] = [self.sim.max_vel[0], 0.0, 0.0]
        self.sim.walk_policy.gait_phase = 0.25
        for _ in range(500):
            self.update()
        self.assertGreater(self.sim._effective_walk_cmd[0], 0.0)
        self.assertGreater(
            abs(self.sim._effective_walk_cmd[0]), self.sim.move_command_threshold
        )

        self.sim.walk_policy.gait_phase = 0.05
        crossed_zero = False
        for _ in range(1000):
            self.update()
            if self.sim._effective_walk_cmd[0] <= 0.0:
                crossed_zero = True
            if np.allclose(self.sim._effective_walk_cmd, self.sim.cmd, atol=1.0e-12):
                break
        self.assertTrue(crossed_zero)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, self.sim.cmd, atol=1.0e-12)

    def test_timeout_keeps_walking_policy_at_zero_command(self):
        self.sim.pending_rl_state = "RL_STAND"
        self.sim._walk_stop_stage = "ZERO_HOLD"
        self.sim._walk_stop_block_reason = "test instability"
        self.sim.switch_total_timeout = 3 * self.sim.control_dt
        self.sim.cmd[:] = [0.2, 0.1, 0.0]
        self.sim._effective_walk_cmd[:] = 0.0
        self.gyro[0] = self.sim.switch_gyro_xy_max
        self.update(3)
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertEqual(self.sim.pending_rl_state, "RL_STAND")
        self.assertEqual(self.sim._walk_stop_stage, "ZERO_HOLD")
        np.testing.assert_allclose(self.sim.cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)

        self.gyro[:] = 0.0
        self.update(int(np.ceil(self.sim.switch_stable_confirm_duration / self.sim.control_dt)))
        self.assertEqual(self.sim.state, "RL_STAND")

    def test_policy_history_and_applied_targets_are_continuous(self):
        previous = np.linspace(-0.5, 0.5, self.sim.walk_policy.n_act).astype(np.float32)
        previous_previous = previous * 0.5
        self.sim.walk_policy.last_actions[:] = previous
        self.sim.walk_policy.last_last_actions[:] = previous_previous
        leg_target = self.sim.stand_pose + np.linspace(-0.03, 0.03, self.sim.nj)
        neck_target = self.sim.neck_default + np.linspace(-0.02, 0.02, self.sim.n_neck)
        self.sim.rl_target = leg_target.copy()
        self.sim.neck_target = neck_target.copy()
        self.sim.torso_cmd[:] = [0.01, 0.1, 0.1, 0.05]

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._complete_rl_switch("RL_STAND")

        np.testing.assert_allclose(self.sim.stand_policy.last_actions, previous)
        np.testing.assert_allclose(
            self.sim.stand_policy.last_last_actions, previous_previous
        )
        for value in (
            self.sim.policy_target_prev,
            self.sim.policy_target_next,
            self.sim.filtered_policy_target,
        ):
            np.testing.assert_allclose(value, leg_target)
        for value in (
            self.sim.neck_policy_target_prev,
            self.sim.neck_policy_target_next,
            self.sim.filtered_neck_target,
        ):
            np.testing.assert_allclose(value, neck_target)
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_torso_cmd, 0.0)
        np.testing.assert_allclose(self.sim.cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)

    def test_switch_to_walk_clears_model_specific_commands(self):
        self.sim.state = "RL_STAND"
        self.sim.policy = self.sim.stand_policy
        self.sim.cmd[:] = [0.2, 0.1, 0.3]
        self.sim._effective_walk_cmd[:] = self.sim.cmd
        self.sim.torso_cmd[:] = [0.01, 0.1, 0.1, 0.05]
        self.sim._effective_torso_cmd[:] = self.sim.torso_cmd

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._complete_rl_switch("RL_WALK")

        self.assertEqual(self.sim.state, "RL_WALK")
        np.testing.assert_allclose(self.sim.cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_torso_cmd, 0.0)

    def test_stand_to_walk_max_squat_command_ramps_to_neutral(self):
        self.set_stand([self.sim.torso_command_min[0], 0.0, 0.0, 0.0])
        start = self.sim._effective_torso_cmd.copy()
        self.request("RL_WALK")
        self.update()

        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertIs(self.sim.policy, self.sim.stand_policy)
        self.assertEqual(self.sim.pending_rl_state, "RL_WALK")
        self.assertEqual(self.sim._stand_to_walk_stage, "RECENTER")
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)
        self.assertGreater(abs(self.sim._effective_torso_cmd[0]), 0.9 * abs(start[0]))
        self.assertAlmostEqual(
            self.sim._stand_to_walk_recenter_duration,
            self.sim.stand_to_walk_recenter_max_duration,
        )

        heights = [float(self.sim._effective_torso_cmd[0])]
        while self.sim._stand_to_walk_stage == "RECENTER":
            self.update()
            heights.append(float(self.sim._effective_torso_cmd[0]))
        self.assertTrue(np.all(np.diff(heights) >= -1.0e-12))
        self.assertAlmostEqual(heights[-1], 0.0)
        self.assertEqual(self.sim.state, "RL_STAND")

    def test_stand_to_walk_head_command_ramps_to_neutral(self):
        self.set_stand()
        self.sim.head_cmd[:] = self.sim.stand_max_head
        start = self.sim.head_cmd.copy()
        self.request("RL_WALK")
        self.update()

        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertEqual(self.sim._stand_to_walk_stage, "RECENTER")
        np.testing.assert_allclose(self.sim.head_cmd, 0.0)
        self.assertGreater(
            np.min(self.sim._active_policy_head_command() / start), 0.9
        )
        self.assertAlmostEqual(
            self.sim._stand_to_walk_recenter_duration,
            self.sim.stand_to_walk_recenter_max_duration,
        )

        peaks = [float(np.max(np.abs(self.sim._active_policy_head_command())))]
        while self.sim._stand_to_walk_stage == "RECENTER":
            self.update()
            peaks.append(float(np.max(np.abs(self.sim._active_policy_head_command()))))
        self.assertTrue(np.all(np.diff(peaks) <= 1.0e-12))
        self.assertAlmostEqual(peaks[-1], 0.0)
        self.assertEqual(self.sim.state, "RL_STAND")

    def test_stand_to_walk_recenter_duration_scales_with_command(self):
        cases = [
            ([0.005, 0.0, 0.0, 0.0], 0.125),
            ([0.0, self.sim.torso_command_max[1] * 0.5, 0.0, 0.0], 0.5),
            ([0.0, 0.0, self.sim.torso_command_min[2], 0.0], 1.0),
            ([self.sim.torso_command_min[0], 0.1, -0.1, 0.05], 1.0),
        ]
        for command, expected_ratio in cases:
            with self.subTest(command=command):
                self.set_stand(command)
                self.request("RL_WALK")
                self.update()
                self.assertAlmostEqual(
                    self.sim._stand_to_walk_recenter_duration,
                    self.sim.stand_to_walk_recenter_max_duration * expected_ratio,
                    places=6,
                )

    def test_stand_to_walk_stability_confirmation_must_be_continuous(self):
        self.set_stand()
        self.sim.stand_to_walk_zero_hold_duration = 0.0
        self.sim.stand_to_walk_stable_confirm_duration = 4 * self.sim.control_dt
        self.request("RL_WALK")
        self.update(3)
        self.assertAlmostEqual(
            self.sim._stand_to_walk_stable_elapsed, 3 * self.sim.control_dt
        )

        self.gyro[0] = self.sim.switch_gyro_xy_max
        self.update()
        self.assertEqual(self.sim._stand_to_walk_stable_elapsed, 0.0)
        self.assertEqual(self.sim.state, "RL_STAND")

        self.gyro[:] = 0.0
        self.update(4)
        self.assertEqual(self.sim.state, "RL_WALK")

    def test_stand_to_walk_cancel_continues_smoothly_to_neutral(self):
        self.set_stand([self.sim.torso_command_min[0], 0.0, 0.0, 0.0])
        self.request("RL_WALK")
        self.update(20)
        before_cancel = self.sim._effective_torso_cmd.copy()

        self.request("RL_STAND")
        self.update()
        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertIsNone(self.sim.pending_rl_state)
        self.assertIs(self.sim.policy, self.sim.stand_policy)
        self.assertLess(
            np.max(np.abs(self.sim._effective_torso_cmd - before_cancel)),
            1.0e-3,
        )
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)

        for _ in range(1000):
            self.update()
            if self.sim._stand_to_walk_stage is None:
                break
        self.assertIsNone(self.sim._stand_to_walk_stage)
        np.testing.assert_allclose(self.sim._effective_torso_cmd, 0.0)

    def test_stand_to_walk_timeout_keeps_standing_neutral(self):
        self.set_stand()
        self.sim.stand_to_walk_zero_hold_duration = 0.0
        self.sim.stand_to_walk_stable_confirm_duration = 2 * self.sim.control_dt
        self.sim.stand_to_walk_total_timeout = 3 * self.sim.control_dt
        self.gyro[0] = self.sim.switch_gyro_xy_max
        self.request("RL_WALK")
        self.update(3)

        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertIs(self.sim.policy, self.sim.stand_policy)
        self.assertEqual(self.sim.pending_rl_state, "RL_WALK")
        self.assertEqual(self.sim._stand_to_walk_stage, "ZERO_HOLD")
        np.testing.assert_allclose(self.sim.cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)
        np.testing.assert_allclose(self.sim._effective_torso_cmd, 0.0)

        self.gyro[:] = 0.0
        self.update(2)
        self.assertEqual(self.sim.state, "RL_WALK")

    def test_stand_to_walk_requires_canonical_encoder_fk_stance(self):
        relative_xy, relative_yaw = self.sim._foot_relative_pose_from_joint_positions(
            self.sim.stand_pose
        )
        np.testing.assert_allclose(relative_xy, self.sim.neutral_foot_relative_xy)
        self.assertAlmostEqual(relative_yaw, self.sim.neutral_foot_relative_yaw)

        # Stable 50 mm narrow-stance IK pose. Its joint error passes the old q-only gate,
        # but encoder FK must keep the standing policy active until the feet are restored.
        narrow_q = np.array(
            [-0.013, -0.109, -0.060, 0.096, 0.037,
             0.013, 0.107, 0.059, -0.095, -0.036]
        )
        self.sim.rl_target = narrow_q.copy()
        common_stable, _ = self.sim._walk_to_stand_stable(
            narrow_q, self.dq, self.quat, self.gyro
        )
        self.assertTrue(common_stable)
        stable, reason = self.sim._stand_to_walk_stable(
            narrow_q, self.dq, self.quat, self.gyro
        )
        self.assertFalse(stable)
        self.assertIn("stance_width_err", reason)

    def test_neutral_stand_to_walk_uses_fast_path(self):
        self.set_stand()
        self.sim.stand_to_walk_zero_hold_duration = 0.0
        self.sim.stand_to_walk_stable_confirm_duration = 2 * self.sim.control_dt
        self.request("RL_WALK")
        self.update()
        self.assertEqual(self.sim._stand_to_walk_stage, "ZERO_HOLD")
        self.assertEqual(self.sim._stand_to_walk_recenter_duration, 0.0)
        self.assertEqual(self.sim.state, "RL_STAND")
        self.update()
        self.assertEqual(self.sim.state, "RL_WALK")

    def test_current_onnx_startup_and_walking_round_trip_meets_latency_bounds(self):
        """Cover the real stand-up plant and current policies, not only ideal q=0 inputs."""
        sim_steps_per_control = max(
            1,
            int(round(self.sim.control_dt / self.sim.model.opt.timestep)),
        )

        def physics_step():
            tau, neck_tau = self.sim.control_step()
            for _ in range(sim_steps_per_control):
                self.sim.data.qfrc_applied[:] = 0.0
                self.sim.data.qfrc_applied[self.sim.v_adr] = tau
                self.sim.data.qfrc_applied[self.sim.neck_v_adr] = neck_tau
                ms.mujoco.mj_step(self.sim.model, self.sim.data)

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim.reset()
            self.sim._cmd_key("1")
            for _ in range(int(np.ceil(4.0 / self.sim.control_dt))):
                physics_step()
                if self.sim.state == "RL_STAND":
                    break
            self.assertEqual(self.sim.state, "RL_STAND")

            self.sim._switch_rl_state("RL_WALK")
            stand_to_walk_steps = 0
            for stand_to_walk_steps in range(int(np.ceil(2.0 / self.sim.control_dt))):
                physics_step()
                if self.sim.state == "RL_WALK":
                    break
            self.assertEqual(self.sim.state, "RL_WALK")
            self.assertLessEqual(
                (stand_to_walk_steps + 1) * self.sim.control_dt,
                2.0,
            )

            self.sim.cmd[:] = [0.25, 0.0, 0.0]
            for _ in range(int(np.ceil(4.0 / self.sim.control_dt))):
                physics_step()
            self.assertEqual(self.sim.state, "RL_WALK")

            self.sim._switch_rl_state("RL_STAND")
            walk_to_stand_steps = 0
            for walk_to_stand_steps in range(int(np.ceil(3.0 / self.sim.control_dt))):
                physics_step()
                if self.sim.state == "RL_STAND":
                    break
            self.assertEqual(self.sim.state, "RL_STAND")
            self.assertLessEqual(
                (walk_to_stand_steps + 1) * self.sim.control_dt,
                3.0,
            )
            self.assertFalse(self.sim._walk_stop_timeout_reported)

    def test_gamepad_centered_stick_does_not_leave_requested_walk_mode(self):
        self.sim.puppeteer.walk_requested = True
        self.gamepad_update(self.gamepad_snapshot())
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertIsNone(self.sim.pending_rl_state)
        np.testing.assert_allclose(self.sim.cmd, 0.0)

    def test_gamepad_normal_lateral_command_clears_policy_deadband(self):
        speed = self.sim.puppeteer.min_lateral_walk_speed
        max_step = self.sim.walk_command_accel_limits[1] * self.sim.control_dt
        steps = int(np.ceil(speed / max_step)) + 2

        for button, direction in ((8, 1.0), (9, -1.0)):
            with self.subTest(button=button):
                self.sim.puppeteer.walk_requested = True
                self.sim.cmd[:] = 0.0
                self.sim._effective_walk_cmd[:] = 0.0
                self.sim._walk_command_safe_zero_active = False
                previous = 0.0

                for _ in range(steps):
                    self.gamepad_update(self.gamepad_snapshot(buttons={button: 1}))
                    current = float(self.sim._effective_walk_cmd[1])
                    self.assertAlmostEqual(self.sim.cmd[1], direction * speed)
                    np.testing.assert_allclose(self.sim.cmd[[0, 2]], 0.0)
                    self.assertGreater(direction * current, 0.0)
                    self.assertGreaterEqual(
                        direction * current + 1.0e-12, direction * previous
                    )
                    self.assertLessEqual(abs(current - previous), max_step + 1.0e-12)
                    previous = current

                self.assertGreater(abs(self.sim.cmd[1]), self.sim.move_command_threshold)
                self.assertAlmostEqual(previous, direction * speed)

    def test_gamepad_start_from_sit_runs_the_same_stand_up_path_as_key_1(self):
        self.sim.state = "SIT"
        self.sim.gamepad_enabled = True
        self.gamepad_update(self.gamepad_snapshot(buttons={11: 1}))

        self.assertEqual(self.sim.state, "STAND_UP")
        self.assertEqual(self.sim.state_time, 0.0)

    def test_gamepad_start_from_walk_uses_the_safe_standing_transition(self):
        self.sim.puppeteer.walk_requested = True
        self.gamepad_update(self.gamepad_snapshot(buttons={11: 1}))

        self.assertFalse(self.sim._puppeteer_walk_requested)
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertEqual(self.sim.pending_rl_state, "RL_STAND")
        self.assertEqual(self.sim._walk_stop_stage, "DECEL")
        self.assertEqual(self.sim._walk_stop_source, "gamepad")

    def test_gamepad_r1_stop_reuses_safe_walk_to_stand_transition(self):
        self.sim.puppeteer.walk_requested = True
        self.gamepad_update(self.gamepad_snapshot(buttons={7: 1}))
        self.gamepad_update(self.gamepad_snapshot())

        self.assertFalse(self.sim._puppeteer_walk_requested)
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertEqual(self.sim.pending_rl_state, "RL_STAND")
        self.assertEqual(self.sim._walk_stop_source, "gamepad")
        self.assertEqual(self.sim._walk_stop_stage, "DECEL")
        np.testing.assert_allclose(self.sim.cmd, 0.0)
        self.assertGreater(np.max(np.abs(self.sim._effective_walk_cmd)), 0.0)

    def test_gamepad_retoggle_walk_cancels_an_active_stop(self):
        self.sim.puppeteer.walk_requested = True
        self.gamepad_update(self.gamepad_snapshot(buttons={7: 1}))
        self.gamepad_update(self.gamepad_snapshot())
        stopped_at = self.sim._effective_walk_cmd.copy()

        self.gamepad_update(self.gamepad_snapshot(axes={1: -1.0}, buttons={7: 1}))
        resumed_at_press = self.sim._effective_walk_cmd.copy()
        self.assertLessEqual(
            np.max(
                np.abs(resumed_at_press - stopped_at)
                - self.sim.walk_command_accel_limits * self.sim.control_dt
            ),
            1.0e-12,
        )
        self.gamepad_update(self.gamepad_snapshot(axes={1: -1.0}))

        self.assertTrue(self.sim._puppeteer_walk_requested)
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertIsNone(self.sim.pending_rl_state)
        self.assertLessEqual(
            np.max(
                np.abs(self.sim._effective_walk_cmd - resumed_at_press)
                - self.sim.walk_command_accel_limits * self.sim.control_dt
            ),
            1.0e-12,
        )

    def test_gamepad_stand_to_walk_preserves_target_and_slews_effective_command(self):
        self.set_stand()
        self.sim.gamepad_enabled = True
        neutral = self.gamepad_snapshot()
        ready_steps = int(
            np.ceil(
                self.sim.stand_to_walk_stable_confirm_duration / self.sim.control_dt
            )
        )
        self.gamepad_update(neutral, ready_steps)

        forward_pressed = self.gamepad_snapshot(axes={1: -1.0}, buttons={7: 1})
        forward_released = self.gamepad_snapshot(axes={1: -1.0})
        self.gamepad_update(forward_pressed)

        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertAlmostEqual(self.sim.cmd[0], self.sim.max_vel[0] * 0.5)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)

        self.gamepad_update(forward_released)
        self.assertGreater(self.sim._effective_walk_cmd[0], 0.0)
        self.assertLessEqual(
            self.sim._effective_walk_cmd[0],
            self.sim.walk_command_accel_limits[0] * self.sim.control_dt + 1.0e-12,
        )

    def test_gamepad_prepositioned_stick_survives_full_stand_recenter(self):
        self.set_stand([0.0, self.sim.torso_command_max[1], 0.0, 0.0])
        self.sim.gamepad_enabled = True
        full_forward_pressed = self.gamepad_snapshot(
            axes={1: -1.0}, buttons={7: 1}
        )
        full_forward_released = self.gamepad_snapshot(axes={1: -1.0})

        self.gamepad_update(full_forward_pressed)
        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertEqual(self.sim._stand_to_walk_stage, "RECENTER")
        self.assertAlmostEqual(self.sim.cmd[0], self.sim.max_vel[0] * 0.5)

        self.gamepad_update(full_forward_released)
        for _ in range(1000):
            if self.sim.state == "RL_WALK":
                break
            self.gamepad_update(full_forward_released)

        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertAlmostEqual(self.sim.cmd[0], self.sim.max_vel[0] * 0.5)
        np.testing.assert_allclose(self.sim._effective_walk_cmd, 0.0)

        self.gamepad_update(full_forward_released)
        expected_step = self.sim.walk_command_accel_limits[0] * self.sim.control_dt
        self.assertAlmostEqual(self.sim._effective_walk_cmd[0], expected_step)

    def test_gamepad_head_input_cannot_override_stand_recenter(self):
        self.set_stand()
        self.sim.gamepad_enabled = True
        look_and_toggle = self.gamepad_snapshot(axes={3: -1.0}, buttons={7: 1})
        look_held = self.gamepad_snapshot(axes={3: -1.0})

        self.gamepad_update(look_held)
        self.assertAlmostEqual(self.sim.head_cmd[1], self.sim.stand_max_head[1])
        self.gamepad_update(look_and_toggle)
        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertEqual(self.sim._stand_to_walk_stage, "RECENTER")
        self.assertGreater(self.sim._active_policy_head_command()[1], 0.0)
        np.testing.assert_allclose(self.sim.head_cmd, 0.0)

        previous = self.sim._active_policy_head_command().copy()
        for _ in range(1000):
            if self.sim.state == "RL_WALK":
                break
            self.gamepad_update(look_held)
            current = self.sim._active_policy_head_command().copy()
            self.assertLessEqual(np.max(np.abs(current)), np.max(np.abs(previous)) + 1.0e-12)
            np.testing.assert_allclose(self.sim.head_cmd, 0.0)
            previous = current

        self.assertEqual(self.sim.state, "RL_WALK")
        np.testing.assert_allclose(self.sim.head_cmd, 0.0)
        self.gamepad_update(look_held)
        self.assertAlmostEqual(self.sim.head_cmd[1], self.sim.walk_max_head[1])

    def test_gamepad_release_requires_posture_inputs_to_return_neutral(self):
        self.sim.puppeteer.posture_rearm_duration = 3 * self.sim.control_dt
        walking = self.gamepad_snapshot(
            axes={1: -1.0}, buttons={7: 1, 8: 1}
        )
        released_with_inputs_held = self.gamepad_snapshot(
            axes={1: -1.0}, buttons={8: 1}
        )

        self.sim.puppeteer.walk_requested = True
        self.gamepad_update(walking)
        self.gamepad_update(released_with_inputs_held)
        self.assertEqual(self.sim.pending_rl_state, "RL_STAND")
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)

        with contextlib.redirect_stdout(io.StringIO()):
            self.sim._complete_rl_switch("RL_STAND")
        self.gamepad_update(released_with_inputs_held, 10)
        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertFalse(self.sim.puppeteer.posture_inputs_armed)
        np.testing.assert_allclose(self.sim.torso_cmd, 0.0)

        self.gamepad_update(self.gamepad_snapshot(), 4)
        self.assertTrue(self.sim.puppeteer.posture_inputs_armed)
        self.gamepad_update(released_with_inputs_held)
        self.assertAlmostEqual(self.sim.torso_cmd[1], self.sim.torso_command_max[1])
        self.assertAlmostEqual(self.sim.torso_cmd[3], self.sim.torso_command_max[3])

    def test_gamepad_cancelled_recenter_is_not_requeued_each_control_step(self):
        self.set_stand([self.sim.torso_command_min[0], 0.0, 0.0, 0.0])
        self.sim.gamepad_enabled = True

        self.gamepad_update(self.gamepad_snapshot(buttons={7: 1}))
        self.gamepad_update(self.gamepad_snapshot())
        self.assertEqual(self.sim._stand_to_walk_stage, "RECENTER")

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.sim._apply_puppeteering_snapshot(
                self.gamepad_snapshot(buttons={7: 1})
            )
            self.sim._update_puppeteering_switch(self.q, self.dq, self.quat, self.gyro)
            self.sim._update_pending_rl_switch(self.q, self.dq, self.quat, self.gyro)
            self.sim._apply_puppeteering_snapshot(self.gamepad_snapshot())
            self.sim._update_puppeteering_switch(self.q, self.dq, self.quat, self.gyro)
            self.sim._update_pending_rl_switch(self.q, self.dq, self.quat, self.gyro)
            for _ in range(10):
                self.sim._apply_puppeteering_snapshot(self.gamepad_snapshot())
                self.sim._update_puppeteering_switch(self.q, self.dq, self.quat, self.gyro)
                self.sim._update_pending_rl_switch(self.q, self.dq, self.quat, self.gyro)

        self.assertTrue(self.sim._stand_to_walk_cancelled)
        self.assertEqual(output.getvalue().count("stand->walk cancelled"), 1)

    def test_gamepad_switch_timeout_keeps_waiting_until_stable(self):
        self.sim.gamepad_enabled = True
        self.sim.puppeteer.walk_requested = False
        self.sim.pending_rl_state = "RL_STAND"
        self.sim._walk_stop_source = "gamepad"
        self.sim._walk_stop_stage = "ZERO_HOLD"
        self.sim._walk_stop_block_reason = "test instability"
        self.sim.switch_total_timeout = 2 * self.sim.control_dt
        self.gyro[0] = self.sim.switch_gyro_xy_max

        neutral = self.gamepad_snapshot()
        self.gamepad_update(neutral, 2)
        self.assertEqual(self.sim.state, "RL_WALK")
        self.assertEqual(self.sim.pending_rl_state, "RL_STAND")
        self.assertIsNone(self.sim._puppeteer_blocked_target)
        self.assertEqual(self.sim._walk_stop_stage, "ZERO_HOLD")

        self.gyro[:] = 0.0
        ready_steps = int(
            np.ceil(self.sim.switch_stable_confirm_duration / self.sim.control_dt)
        )
        self.gamepad_update(neutral, ready_steps)
        self.assertEqual(self.sim.state, "RL_STAND")
        self.assertIsNone(self.sim.pending_rl_state)
        self.assertIsNone(self.sim._walk_stop_stage)


if __name__ == "__main__":
    unittest.main()
