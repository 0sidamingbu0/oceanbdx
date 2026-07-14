#!/usr/bin/env python3
"""Linux joystick input and the paper-aligned BDX puppeteering mapping."""

from __future__ import annotations

import errno
import os
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np


_JS_EVENT = struct.Struct("IhBB")
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_AXIS = 0x02
_JS_EVENT_INIT = 0x80


@dataclass(frozen=True)
class GamepadSnapshot:
    axes: np.ndarray
    buttons: np.ndarray
    connected: bool = True

    @staticmethod
    def zero(connected: bool = False) -> "GamepadSnapshot":
        return GamepadSnapshot(
            axes=np.zeros(8, dtype=np.float32),
            buttons=np.zeros(16, dtype=np.int8),
            connected=connected,
        )


@dataclass(frozen=True)
class PuppeteeringCommand:
    walk_requested: bool
    stand_requested: bool
    start_requested: bool
    walk_command: np.ndarray
    torso_command: np.ndarray
    head_command: np.ndarray
    full_speed: bool
    connected: bool


class LinuxJoystick:
    """Read ``/dev/input/js*`` on a background thread and publish snapshots."""

    def __init__(self, device: str, reconnect_interval_s: float = 1.0):
        self.device = device
        self.reconnect_interval_s = max(0.1, float(reconnect_interval_s))
        self._lock = threading.Lock()
        self._snapshot = GamepadSnapshot.zero()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._publish(GamepadSnapshot.zero())

    def snapshot(self) -> GamepadSnapshot:
        with self._lock:
            value = self._snapshot
            return GamepadSnapshot(value.axes.copy(), value.buttons.copy(), value.connected)

    def _publish(self, snapshot: GamepadSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def _read_loop(self) -> None:
        fd = -1
        axes = np.zeros(8, dtype=np.float32)
        buttons = np.zeros(16, dtype=np.int8)
        while self._running:
            if fd < 0:
                try:
                    fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
                    axes.fill(0.0)
                    buttons.fill(0)
                    self._publish(GamepadSnapshot(axes.copy(), buttons.copy(), True))
                    print(f"[gamepad] connected: {self.device}")
                except OSError:
                    self._publish(GamepadSnapshot.zero())
                    time.sleep(self.reconnect_interval_s)
                    continue

            try:
                payload = os.read(fd, _JS_EVENT.size)
                if len(payload) != _JS_EVENT.size:
                    raise OSError(errno.ENODEV, "short joystick read")
                _, raw_value, event_type, number = _JS_EVENT.unpack(payload)
                event_type &= ~_JS_EVENT_INIT
                if event_type == _JS_EVENT_AXIS and number < axes.size:
                    axes[number] = np.clip(raw_value / 32767.0, -1.0, 1.0)
                elif event_type == _JS_EVENT_BUTTON and number < buttons.size:
                    buttons[number] = 1 if raw_value else 0
                self._publish(GamepadSnapshot(axes.copy(), buttons.copy(), True))
            except BlockingIOError:
                time.sleep(0.002)
            except OSError:
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = -1
                axes.fill(0.0)
                buttons.fill(0)
                self._publish(GamepadSnapshot.zero())
                print(f"[gamepad] disconnected: {self.device}; requesting standing mode")

        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


class PuppeteeringMapper:
    """Map a two-stick controller to the Appendix C standing/walking commands.

    ``toggle`` follows Table VII. ``hold`` is an optional deadman adaptation for
    deployments that should walk only while R1 remains pressed.
    """

    def __init__(
        self,
        cfg: dict,
        max_velocity,
        torso_min,
        torso_max,
        max_head,
        walking_max_head=None,
    ):
        self.cfg = cfg
        self.max_velocity = np.asarray(max_velocity, dtype=np.float32)
        self.torso_min = np.asarray(torso_min, dtype=np.float32)
        self.torso_max = np.asarray(torso_max, dtype=np.float32)
        self.max_head = np.asarray(max_head, dtype=np.float32)
        self.walking_max_head = np.asarray(
            max_head if walking_max_head is None else walking_max_head,
            dtype=np.float32,
        )
        if self.max_head.shape != (4,) or self.walking_max_head.shape != (4,):
            raise ValueError("standing and walking head limits must each contain four values")

        self.deadzone = float(cfg.get("axis_deadzone", 0.08))
        self.walk_button_behavior = str(cfg.get("walk_button_behavior", "toggle")).lower()
        if self.walk_button_behavior not in ("toggle", "hold"):
            raise ValueError("puppeteering.walk_button_behavior must be 'toggle' or 'hold'")
        self.r1_hold_duration = max(0.05, float(cfg.get("r1_hold_duration_s", 0.35)))
        self.normal_walk_gain = float(cfg.get("normal_walk_gain", 0.5))
        self.full_walk_gain = float(cfg.get("full_walk_gain", 1.0))
        self.hold_walk_gain = float(cfg.get("hold_walk_gain", self.normal_walk_gain))
        self.min_lateral_walk_speed = float(
            cfg.get("min_lateral_walk_speed", 0.0)
        )
        self.posture_rearm_duration = max(
            0.0, float(cfg.get("posture_rearm_duration_s", 0.15))
        )
        self.gaze_torso_threshold = float(cfg.get("gaze_torso_threshold", 0.75))
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("puppeteering.axis_deadzone must be in [0, 1)")
        if not 0.0 <= self.gaze_torso_threshold < 1.0:
            raise ValueError("puppeteering.gaze_torso_threshold must be in [0, 1)")
        if min(self.normal_walk_gain, self.full_walk_gain, self.hold_walk_gain) < 0.0:
            raise ValueError("puppeteering walk gains must be non-negative")
        if not 0.0 <= self.min_lateral_walk_speed <= float(self.max_velocity[1]):
            raise ValueError(
                "puppeteering.min_lateral_walk_speed must be within [0, max_vy]"
            )

        mapping = cfg.get("mapping", {})
        self.axis_left_x = int(mapping.get("axis_left_x", 0))
        self.axis_left_y = int(mapping.get("axis_left_y", 1))
        self.axis_right_x = int(mapping.get("axis_right_x", 2))
        self.axis_right_y = int(mapping.get("axis_right_y", 3))
        self.axis_dpad_x = int(mapping.get("axis_dpad_x", 6))
        self.axis_dpad_y = int(mapping.get("axis_dpad_y", 7))
        self.button_a = int(mapping.get("button_a", 0))
        self.button_r1 = int(mapping.get("button_r1", 7))
        self.button_l2 = int(mapping.get("button_l2", 8))
        self.button_r2 = int(mapping.get("button_r2", 9))
        self.button_start = int(mapping.get("button_start", 11))

        self.walk_requested = False
        self._r1_was_pressed = False
        self._r1_press_elapsed = 0.0
        self._r1_long_press = False
        self._r1_gesture_cancelled = False
        self._r1_started_walk = False
        self._stand_was_pressed = False
        self._start_was_pressed = False
        self._posture_inputs_armed = False
        self._posture_neutral_elapsed = 0.0

    def reset(self) -> None:
        self.walk_requested = False
        self._r1_was_pressed = False
        self._r1_press_elapsed = 0.0
        self._r1_long_press = False
        self._r1_gesture_cancelled = False
        self._r1_started_walk = False
        self._stand_was_pressed = False
        self._start_was_pressed = False
        self._posture_inputs_armed = False
        self._posture_neutral_elapsed = 0.0

    @property
    def posture_inputs_armed(self) -> bool:
        return self._posture_inputs_armed

    @staticmethod
    def _read(values: np.ndarray, index: int) -> float:
        return float(values[index]) if 0 <= index < values.size else 0.0

    @staticmethod
    def _pressed(values: np.ndarray, index: int) -> bool:
        return bool(values[index]) if 0 <= index < values.size else False

    def _axis(self, value: float) -> float:
        value = float(np.clip(value, -1.0, 1.0))
        magnitude = abs(value)
        if magnitude <= self.deadzone:
            return 0.0
        return np.sign(value) * (magnitude - self.deadzone) / (1.0 - self.deadzone)

    @staticmethod
    def _signed_range(value: float, lower: float, upper: float) -> float:
        return value * (upper if value >= 0.0 else abs(lower))

    def _standing_gaze(self, value: float) -> tuple[float, float]:
        """Split standing gaze input into local-head and torso-extension components."""
        magnitude = abs(value)
        if magnitude == 0.0:
            return 0.0, 0.0
        if self.gaze_torso_threshold <= 1.0e-6:
            return float(np.sign(value)), value
        if magnitude <= self.gaze_torso_threshold:
            return value / self.gaze_torso_threshold, 0.0
        scaled = (magnitude - self.gaze_torso_threshold) / (1.0 - self.gaze_torso_threshold)
        return float(np.sign(value)), float(np.sign(value) * scaled)

    def _update_mode(
        self, r1_pressed: bool, stand_pressed: bool, dt: float
    ) -> tuple[bool, bool]:
        stand_rising = stand_pressed and not self._stand_was_pressed
        if self.walk_button_behavior == "hold":
            if r1_pressed and not self._r1_was_pressed:
                self._r1_gesture_cancelled = stand_pressed
                self._r1_started_walk = False
            if not r1_pressed:
                self._r1_gesture_cancelled = False
            self.walk_requested = r1_pressed and not self._r1_gesture_cancelled
            self._r1_press_elapsed = 0.0
            self._r1_long_press = False
            full_speed = False
        else:
            if r1_pressed and not self._r1_was_pressed:
                self._r1_press_elapsed = 0.0
                self._r1_long_press = False
                self._r1_gesture_cancelled = stand_pressed
                # Starting to walk is unambiguous while standing, so consume the
                # press edge immediately. This preserves a stick command that is
                # already active when R1 is pressed. While walking, the release edge
                # still distinguishes a short stop press from a full-speed hold.
                self._r1_started_walk = not self.walk_requested and not stand_pressed
                if self._r1_started_walk:
                    self.walk_requested = True
            if r1_pressed:
                self._r1_press_elapsed += dt
                if self._r1_press_elapsed >= self.r1_hold_duration:
                    self._r1_long_press = True
            if not r1_pressed and self._r1_was_pressed:
                if (
                    not self._r1_long_press
                    and not self._r1_gesture_cancelled
                    and not stand_pressed
                    and not self._r1_started_walk
                ):
                    self.walk_requested = not self.walk_requested
                self._r1_press_elapsed = 0.0
                self._r1_gesture_cancelled = False
                self._r1_started_walk = False
            full_speed = (
                r1_pressed
                and self._r1_long_press
                and not self._r1_gesture_cancelled
                and self.walk_requested
            )

        # A/START always win. They also consume an R1 gesture already in progress
        # so releasing R1 cannot immediately toggle walking back on.
        if stand_rising:
            self.walk_requested = False
            if r1_pressed or self._r1_was_pressed:
                self._r1_gesture_cancelled = True
        if stand_pressed:
            self.walk_requested = False
            full_speed = False

        self._r1_was_pressed = r1_pressed
        self._stand_was_pressed = stand_pressed
        return full_speed, stand_rising

    def update(
        self,
        snapshot: GamepadSnapshot,
        dt: float,
        active_walking: bool | None = None,
    ) -> PuppeteeringCommand:
        if not snapshot.connected:
            self.reset()
            return PuppeteeringCommand(
                walk_requested=False,
                stand_requested=False,
                start_requested=False,
                walk_command=np.zeros(3, dtype=np.float32),
                torso_command=np.zeros(4, dtype=np.float32),
                head_command=np.zeros(4, dtype=np.float32),
                full_speed=False,
                connected=False,
            )

        axes = np.asarray(snapshot.axes)
        buttons = np.asarray(snapshot.buttons)
        r1_pressed = self._pressed(buttons, self.button_r1)
        a_pressed = self._pressed(buttons, self.button_a)
        start_pressed = self._pressed(buttons, self.button_start)
        start_requested = start_pressed and not self._start_was_pressed
        full_speed, stand_requested = self._update_mode(
            r1_pressed, a_pressed or start_pressed, float(dt)
        )
        self._start_was_pressed = start_pressed

        left_x = self._axis(self._read(axes, self.axis_left_x))
        left_up = -self._axis(self._read(axes, self.axis_left_y))
        right_x = self._axis(self._read(axes, self.axis_right_x))
        right_up = -self._axis(self._read(axes, self.axis_right_y))
        dpad_x = self._axis(self._read(axes, self.axis_dpad_x))
        dpad_up = -self._axis(self._read(axes, self.axis_dpad_y))
        l2 = 1.0 if self._pressed(buttons, self.button_l2) else 0.0
        r2 = 1.0 if self._pressed(buttons, self.button_r2) else 0.0
        controls_are_walking = (
            self.walk_requested if active_walking is None else bool(active_walking)
        )
        head_limits = self.walking_max_head if controls_are_walking else self.max_head

        head = np.zeros(4, dtype=np.float32)
        head[0] = dpad_up * head_limits[0]
        head[3] = -dpad_x * head_limits[3]
        torso = np.zeros(4, dtype=np.float32)
        walk = np.zeros(3, dtype=np.float32)

        gaze_pitch = right_up
        gaze_yaw = -right_x
        if controls_are_walking:
            # Left stick and triggers have different meanings in walking and standing.
            # Disarm their standing interpretation until they have returned to neutral
            # after the walking policy has actually exited.
            self._posture_inputs_armed = False
            self._posture_neutral_elapsed = 0.0
            head[1] = gaze_pitch * head_limits[1]
            head[2] = gaze_yaw * head_limits[2]
        else:
            head_gaze_pitch, torso_gaze_pitch = self._standing_gaze(gaze_pitch)
            head_gaze_yaw, torso_gaze_yaw = self._standing_gaze(gaze_yaw)
            posture_inputs_neutral = (
                left_x == 0.0 and left_up == 0.0 and l2 == 0.0 and r2 == 0.0
            )
            if not self._posture_inputs_armed:
                if posture_inputs_neutral:
                    self._posture_neutral_elapsed += max(0.0, float(dt))
                    if self._posture_neutral_elapsed >= self.posture_rearm_duration:
                        self._posture_inputs_armed = True
                else:
                    self._posture_neutral_elapsed = 0.0

            posture_left_x = left_x if self._posture_inputs_armed else 0.0
            posture_left_up = left_up if self._posture_inputs_armed else 0.0
            posture_l2 = l2 if self._posture_inputs_armed else 0.0
            posture_r2 = r2 if self._posture_inputs_armed else 0.0

            if posture_left_up >= 0.0:
                torso[1] = posture_left_up * self.torso_max[1]
            else:
                torso[0] = (-posture_left_up) * self.torso_min[0]
            torso[2] = self._signed_range(
                -posture_left_x, self.torso_min[2], self.torso_max[2]
            )
            torso[3] = self._signed_range(
                posture_l2 - posture_r2, self.torso_min[3], self.torso_max[3]
            )

            torso[1] += self._signed_range(
                torso_gaze_pitch, self.torso_min[1], self.torso_max[1]
            )
            torso[2] += self._signed_range(
                torso_gaze_yaw, self.torso_min[2], self.torso_max[2]
            )
            torso[:] = np.clip(torso, self.torso_min, self.torso_max)

            # Left-stick posture control counter-rotates the head to preserve gaze. The
            # right stick controls gaze and adds torso rotation only near the neck limit.
            posture_pitch = (
                posture_left_up * self.torso_max[1] if posture_left_up >= 0.0 else 0.0
            )
            posture_yaw = self._signed_range(
                -posture_left_x, self.torso_min[2], self.torso_max[2]
            )
            posture_roll = self._signed_range(
                posture_l2 - posture_r2, self.torso_min[3], self.torso_max[3]
            )
            head[1] = head_gaze_pitch * head_limits[1] - posture_pitch
            head[2] = head_gaze_yaw * head_limits[2] - posture_yaw
            head[3] -= posture_roll

        if self.walk_requested:
            gain = (
                self.hold_walk_gain
                if self.walk_button_behavior == "hold"
                else self.full_walk_gain if full_speed else self.normal_walk_gain
            )
            walk[0] = left_up * self.max_velocity[0] * gain
            lateral_input = l2 - r2
            lateral_speed = self.max_velocity[1] * gain
            if lateral_input != 0.0 and gain > 0.0:
                lateral_speed = max(lateral_speed, self.min_lateral_walk_speed)
            walk[1] = lateral_input * lateral_speed
            walk[2] = -left_x * self.max_velocity[2] * gain

        head[:] = np.clip(head, -head_limits, head_limits)
        walk[:] = np.clip(walk, -self.max_velocity, self.max_velocity)
        return PuppeteeringCommand(
            walk_requested=self.walk_requested,
            stand_requested=stand_requested,
            start_requested=start_requested,
            walk_command=walk,
            torso_command=torso,
            head_command=head,
            full_speed=full_speed,
            connected=True,
        )
