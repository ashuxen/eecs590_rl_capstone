"""SmartInsert — Cable insertion policy for the AIC challenge.

HOW THIS POLICY WORKS (beginner-friendly overview)
====================================================

The AIC challenge asks a UR5e robot arm to insert a cable connector (SFP or SC
type) into the correct port on a task board.  The robot already holds the cable;
our job is to move the gripper so the plug tip lines up with the port and slides in.

This policy has TWO operating modes:

  MODE 1 — TF mode  (ground_truth:=true, for development)
    The simulator broadcasts the exact 3D positions of every port and plug via
    the ROS 2 "TF" (Transform) system.  We use that perfect knowledge to:
      1.  Smoothly approach a point above/in-front-of the port.
      2.  Descend along the port's insertion axis until the plug seats.
      3.  Hold still while the scoring system verifies insertion.

  MODE 2 — Blind descent  (ground_truth:=false, for evaluation)
    Ground-truth poses are NOT available.  We start from wherever the gripper
    currently is and slowly lower, using force/torque feedback to detect contact
    and make small corrections.

KEY CONCEPTS FOR BEGINNERS
============================

*  TF (Transform library):  ROS 2 maintains a tree of coordinate frames.  Each
   link of the robot, each camera, and (when ground_truth is on) each object in
   the sim has a frame.  We can ask "where is frame A relative to frame B?" at
   any time.

*  Quaternion:  A 4-number representation of 3-D rotation (w, x, y, z).
   More robust than Euler angles; used everywhere in ROS.

*  SLERP (Spherical Linear Interpolation):  Smoothly blends between two
   orientations.  We use it to gradually rotate the gripper from its current
   orientation to the orientation that aligns the plug with the port.

*  Insertion axis:  The direction you push the connector to insert it.  For SFP
   ports on NIC cards this is roughly straight down (world -Z).  For SC fiber
   ports it is roughly horizontal.  We compute this from the port's TF
   orientation so the same code works for any port.

*  Integrator (I-term):  The cable is flexible, so even when the gripper is
   aimed at the port, the plug tip drifts sideways.  We accumulate the lateral
   error over time and apply a small correction — this is a classic integral
   controller from control theory.
"""

import numpy as np
from typing import Optional

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from smart_insert.data_collector import DataCollector
from smart_insert.perception.v1_simple import detect_port_v1
from smart_insert.perception.v2_learned import detect_port_v2

QuaternionTuple = tuple[float, float, float, float]

# ---------------------------------------------------------------------------
#  Force-safety constants  (plan Section 15.2: "Hard-coded, Non-negotiable")
# ---------------------------------------------------------------------------
FORCE_Z_LIMIT = 12.0        # Newtons — retract if Z force exceeds this (penalty at 20N)
FORCE_LATERAL_LIMIT = 8.0   # Newtons — halt if lateral force exceeds this
FORCE_CHANGE_LIMIT = 8.0    # Newtons — react if force *changes* by this much
MAX_FORCE_PAUSES = 5        # stop descent after this many consecutive over-force events

# Retry parameters
MAX_INSERTION_ATTEMPTS = 3
RETRY_MIN_TIME = 25.0       # seconds — need at least this much time remaining to retry
RETRACT_HEIGHT = 0.06       # metres — retract this far along insertion axis before retry
RETRY_SETTLE_STEPS = 20     # steps to settle after retraction

# ---------------------------------------------------------------------------
#  Insertion phase enum + phase-adaptive impedance profiles
#
#  Innovation: instead of a single impedance throughout insertion, each phase
#  has its own stiffness/damping/force profile optimized for that contact regime.
#  Transitions are driven by real-time contact-state classification from F/T.
# ---------------------------------------------------------------------------
PHASE_FREE_SPACE = 0    # far from board, fast approach
PHASE_NEAR_CONTACT = 1  # close to board, prepare for contact
PHASE_ALIGNMENT = 2     # contact detected, funnel toward port opening
PHASE_INSERTION = 3     # plug entering port, push along axis
PHASE_SEATED = 4        # insertion complete, hold

PHASE_IMPEDANCE = {
    PHASE_FREE_SPACE: {
        "K": np.array([200.0, 200.0, 100.0, 60.0, 60.0, 60.0]),
        "D": np.array([60.0, 60.0, 40.0, 25.0, 25.0, 25.0]),
    },
    PHASE_NEAR_CONTACT: {
        "K": np.array([120.0, 120.0, 50.0, 40.0, 40.0, 40.0]),
        "D": np.array([55.0, 55.0, 35.0, 22.0, 22.0, 22.0]),
    },
    PHASE_ALIGNMENT: {
        "K": np.array([40.0, 40.0, 25.0, 30.0, 30.0, 30.0]),
        "D": np.array([45.0, 45.0, 30.0, 20.0, 20.0, 20.0]),
    },
    PHASE_INSERTION: {
        "K": np.array([60.0, 60.0, 80.0, 40.0, 40.0, 40.0]),
        "D": np.array([50.0, 50.0, 40.0, 22.0, 22.0, 22.0]),
    },
    PHASE_SEATED: {
        "K": np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        "D": np.array([60.0, 60.0, 50.0, 30.0, 30.0, 30.0]),
    },
}

# Force-funnel parameters
FUNNEL_CONTACT_THRESHOLD = 3.0   # N — lateral force above baseline = board contact
FUNNEL_INSERTION_THRESHOLD = 5.0 # N — axial force spike = plug entering port
FUNNEL_GAIN = 0.0004             # m/N — how aggressively to move opposite to lateral force
FUNNEL_MAX_CORRECTION = 0.004    # m — max single-step lateral correction
FUNNEL_SEATED_Z_THRESHOLD = 2.0  # N — low axial force after insertion = seated

# Workspace bounds in base_link (derived from training data port positions + margin)
WORKSPACE_MIN = np.array([-0.55, 0.10, -0.02])
WORKSPACE_MAX = np.array([-0.25, 0.40, 0.35])

# TCP-to-plug offsets measured from training data (per connector type).
# Two offsets per type: the cable swings during approach, so the offset
# is very different at the start (plug hanging) vs after settling.
#   APPROACH — used while position_fraction < 1 (gripper still moving to port)
#   SETTLED  — used during descent/hold (cable taut, plug directly below TCP)
PLUG_OFFSET_SFP_APPROACH = np.array([0.0, -0.021, 0.054])   # step 0, plug hanging
PLUG_OFFSET_SFP_SETTLED  = np.array([0.0,  0.000, 0.056])   # step 100+, cable taut
PLUG_OFFSET_SC_APPROACH  = np.array([0.0, -0.012, 0.014])
PLUG_OFFSET_SC_SETTLED   = np.array([0.0, -0.005, 0.020])

# Wrist clearance: for truly horizontal insertion the wrist_3_link extends
# ~10 cm above TCP and can collide with the enclosure ceiling.  During
# approach and early descent we lower the gripper target by this amount;
# it is ramped to zero as z_offset → 0 so the plug still aligns for insertion.
# NOTE: On this task board, SC ports actually insert DOWNWARD (same as SFP),
# so this clearance only activates for ports with abs(z_axis[2]) < 0.5.
SC_WRIST_CLEARANCE_Z = -0.025  # 2.5 cm lower during approach (safety only)
SC_APPROACH_TILT_DEG = 10.0    # degrees, for any truly horizontal port

# Residual action scaling (per ResiP paper Section III-C)
RESIDUAL_ALPHA = 0.1


class SmartInsert(Policy):
    """Cable insertion with axis-aware TF mode and blind-descent fallback."""

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)

        # Task metadata (filled in each call to insert_cable)
        self._task: Optional[Task] = None
        self._start_time_ns: int = 0
        self._time_budget: float = 170.0  # seconds of sim-time we allow ourselves
        self._current_attempt: int = 1

        # 3-D integrator for lateral cable-drift correction (see _calc_gripper_pose)
        self._tip_error_integrator = np.array([0.0, 0.0, 0.0])
        self._max_integrator_windup: float = 0.05  # clamp each axis
        self._i_gain: float = 0.15                  # how aggressively we correct drift

        # Training data collector (activate with AIC_COLLECT_DATA=1)
        self._collector = DataCollector(logger=self.get_logger())

        # Optional YD-RRL residual (Phase 2 per RL_PIPELINE_PLAN.md)
        self._residual_mlp = None
        if __import__("os").environ.get("AIC_USE_YD_RESIDUAL", "").strip().lower() in ("1", "true", "yes"):
            self._residual_mlp = self._load_residual_mlp()
            if self._residual_mlp is not None:
                self.get_logger().info("YD-RRL residual enabled (Phase 2)")
            else:
                self.get_logger().warn("AIC_USE_YD_RESIDUAL=1 but residual MLP failed to load — running base only")

        # Residual training data collection (AIC_COLLECT_RESIDUAL_DATA=1, run with ground_truth:=true)
        self._collect_residual = __import__("os").environ.get("AIC_COLLECT_RESIDUAL_DATA", "").strip() == "1"
        self._residual_steps: list = []
        self._residual_episode_num = 0
        self._residual_data_dir = __import__("pathlib").Path(__file__).resolve().parents[5] / "rl" / "residual_data"
        if not self._residual_data_dir.exists():
            self._residual_data_dir = __import__("pathlib").Path.home() / "rl" / "residual_data"

        # DAgger collection mode (AIC_DAGGER_COLLECT=1, run with ground_truth:=true)
        self._dagger_collect = __import__("os").environ.get("AIC_DAGGER_COLLECT", "").strip() == "1"
        self._dagger_steps: list = []
        self._dagger_round = int(__import__("os").environ.get("AIC_DAGGER_ROUND", "1"))
        self._dagger_data_dir = __import__("pathlib").Path.home() / "rl" / "dagger_data" / f"round_{self._dagger_round:02d}"
        self._dagger_port_frame: str = ""
        self._dagger_plug_frame: str = ""
        if self._dagger_collect:
            self.get_logger().info(f"DAgger collection enabled (round {self._dagger_round})")

        # PPO training mode (AIC_PPO_TRAIN=1, run with ground_truth:=true)
        self._ppo_train = __import__("os").environ.get("AIC_PPO_TRAIN", "").strip() == "1"
        self._ppo_actor = None
        self._ppo_critic = None
        self._ppo_trajectory: list = []
        self._ppo_prev_plug_port_dist: float = 0.0
        self._ppo_port_frame: str = ""
        self._ppo_plug_frame: str = ""
        self._ppo_iter = int(__import__("os").environ.get("AIC_PPO_ITER", "0"))
        self._ppo_data_dir = __import__("pathlib").Path.home() / "rl" / "ppo_training_data" / f"iter_{self._ppo_iter:03d}"
        self._ppo_episode_num = 0
        if self._ppo_train:
            self._ppo_data_dir.mkdir(parents=True, exist_ok=True)
            existing = list(self._ppo_data_dir.glob("episode_*.npz"))
            self._ppo_episode_num = len(existing)
            self._init_ppo_networks()

        # Gaussian HMM phase estimator (replaces hard-threshold classifier)
        self._phase_hmm = None
        try:
            import sys as _sys
            _rl = __import__("pathlib").Path.home() / "rl"
            if str(_rl) not in _sys.path:
                _sys.path.insert(0, str(_rl))
            from training.phase_belief import PhaseGaussianHMM
            hmm_path = _rl / "yd_rrl_checkpoints" / "phase_hmm_params.npz"
            if hmm_path.exists():
                self._phase_hmm = PhaseGaussianHMM.from_file(str(hmm_path))
                self.get_logger().info(f"Gaussian HMM phase estimator loaded from {hmm_path}")
            else:
                self.get_logger().info("HMM params not found — using threshold classifier fallback")
        except Exception as e:
            self.get_logger().warn(f"HMM phase estimator failed to load: {e}")

        self.get_logger().info("SmartInsert policy initialized")

    # ------------------------------------------------------------------
    #  PPO Training Network Initialization
    # ------------------------------------------------------------------

    def _init_ppo_networks(self):
        """Load PPO actor + critic for stochastic on-policy data collection."""
        try:
            import sys
            import torch
            from pathlib import Path
            rl = Path.home() / "rl"
            if str(rl) not in sys.path:
                sys.path.insert(0, str(rl))
            from training.ppo_residual import PPOActor, PPOCritic

            device = torch.device("cpu")
            self._ppo_actor = PPOActor().to(device)
            self._ppo_critic = PPOCritic().to(device)

            # Try to load existing PPO weights (from previous iteration)
            ppo_dir = rl / "yd_rrl_checkpoints" / "ppo_resip"
            if (ppo_dir / "ppo_actor.pt").exists():
                state = torch.load(ppo_dir / "ppo_actor.pt", map_location=device, weights_only=True)
                self._ppo_actor.load_state_dict(state)
                self.get_logger().info(f"PPO actor loaded from {ppo_dir}")
            else:
                # Warm-start from DAgger checkpoint
                dagger_dir = rl / "yd_rrl_checkpoints" / "dagger_r3"
                if (dagger_dir / "residual_mlp.pt").exists():
                    dagger_state = torch.load(
                        dagger_dir / "residual_mlp.pt", map_location=device, weights_only=True,
                    )
                    DAGGER_TO_PPO = {
                        "fc.0.weight": "net.0.weight", "fc.0.bias": "net.0.bias",
                        "fc.2.weight": "net.2.weight", "fc.2.bias": "net.2.bias",
                        "fc.4.weight": "net.4.weight", "fc.4.bias": "net.4.bias",
                        "fc.6.weight": "mean_head.weight", "fc.6.bias": "mean_head.bias",
                    }
                    actor_state = self._ppo_actor.state_dict()
                    mapped = 0
                    for dk, ak in DAGGER_TO_PPO.items():
                        if dk in dagger_state and ak in actor_state:
                            if dagger_state[dk].shape == actor_state[ak].shape:
                                actor_state[ak] = dagger_state[dk]
                                mapped += 1
                    self._ppo_actor.load_state_dict(actor_state)
                    self.get_logger().info(f"PPO actor warm-started from DAgger ({mapped}/8 layers)")

            if (ppo_dir / "ppo_critic.pt").exists():
                state = torch.load(ppo_dir / "ppo_critic.pt", map_location=device, weights_only=True)
                self._ppo_critic.load_state_dict(state)
                self.get_logger().info(f"PPO critic loaded from {ppo_dir}")

            self._ppo_actor.eval()
            self._ppo_critic.eval()
            self.get_logger().info("PPO training mode: stochastic policy active")
        except Exception as e:
            self.get_logger().error(f"PPO network init failed: {e}")
            self._ppo_train = False

    # ==================================================================
    #  PUBLIC ENTRY POINT — called by the aic_model framework
    # ==================================================================

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Main entry: decide which mode to use and run the insertion."""
        import sys
        print(f"[SmartInsert] insert_cable() CALLED — task={task}", flush=True)
        sys.stdout.flush()
        self.get_logger().info(f"SmartInsert.insert_cable() task: {task}")
        self._task = task
        self._start_time_ns = self.time_now().nanoseconds
        self._tip_error_integrator = np.array([0.0, 0.0, 0.0])

        # The framework tells us how long we have; keep a 10-s safety margin.
        if task.time_limit > 0:
            self._time_budget = float(task.time_limit) - 10.0
        self.get_logger().info(
            f"Time budget: {self._time_budget:.1f}s  (task limit={task.time_limit}s)"
        )

        # Build the TF frame names from the task description.
        # e.g.  "task_board/nic_card_mount_0/sfp_port_0_link"
        #        "cable_0/sfp_tip_link"
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        self.get_logger().info(f"Port frame : {port_frame}")
        self.get_logger().info(f"Cable frame: {cable_tip_frame}")

        # Start data collection for this episode.
        self._collector.start_episode(task)

        # PPO training: init trajectory buffer and store TF frame names
        self._current_attempt = 1
        if self._ppo_train:
            self._ppo_trajectory = []
            self._ppo_prev_plug_port_dist = 0.0
            self._ppo_port_frame = port_frame
            self._ppo_plug_frame = cable_tip_frame

        # Try to find the ground-truth TF frames (only available when
        # the eval container was started with ground_truth:=true).
        print(f"[SmartInsert] Checking for port TF: {port_frame}", flush=True)
        has_port_tf = self._wait_for_tf("base_link", port_frame, timeout_sec=5.0)
        print(f"[SmartInsert] has_port_tf={has_port_tf}", flush=True)
        if has_port_tf:
            self._wait_for_tf("base_link", cable_tip_frame, timeout_sec=3.0)

        force_perception = __import__("os").environ.get("AIC_FORCE_PERCEPTION", "0") == "1"

        if has_port_tf:
            if self._dagger_collect:
                self.get_logger().info(
                    "DAgger mode: TF available, forcing perception path for data collection"
                )
                self._dagger_port_frame = port_frame
                self._dagger_plug_frame = cable_tip_frame
                self._dagger_steps = []
                # Fall through to perception detection below
            elif force_perception:
                self.get_logger().info(
                    "AIC_FORCE_PERCEPTION=1: TF available but using perception path for validation"
                )
                # Fall through to perception detection below
            else:
                self.get_logger().info("Ground-truth TF available — using TF mode")
                if self._collect_residual:
                    self._residual_steps = []
                result = self._run_with_tf(task, get_observation, move_robot, send_feedback)
                if self._collect_residual and self._residual_steps:
                    self._save_residual_episode()
                if self._ppo_train:
                    self.get_logger().info(
                        f"PPO post-TF: trajectory_len={len(self._ppo_trajectory)}, result={result}"
                    )
                    if self._ppo_trajectory:
                        self._save_ppo_trajectory(success=result)
                    else:
                        self.get_logger().warn("PPO trajectory is EMPTY — no data collected!")
                self._collector.finish_episode(success=result)
                return result

        # No ground truth (or DAgger mode): try V2 → V1 → blind descent
        perception_version = __import__("os").environ.get(
            "AIC_PERCEPTION_VERSION", "v2"
        ).strip().lower()
        print(f"[SmartInsert] Perception version: {perception_version}", flush=True)

        port_result = None

        if perception_version == "v2":
            print("[SmartInsert] Calling _detect_with_multi_obs (V2)...", flush=True)
            port_result = self._detect_with_multi_obs(
                task, get_observation, n_obs=3, delay=0.15,
            )
            print(f"[SmartInsert] V2 result: {port_result is not None}", flush=True)

        if port_result is None:
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.3)
                obs = get_observation()
            if obs is not None:
                port_result = detect_port_v1(
                    obs, task, self._parent_node._tf_buffer, self.get_logger()
                )

        if port_result is not None:
            port_transform, port_z_axis = port_result
            print(
                f"[SmartInsert] PERCEPTION OK — port at "
                f"[{port_transform.translation.x:.4f}, "
                f"{port_transform.translation.y:.4f}, "
                f"{port_transform.translation.z:.4f}]  axis={port_z_axis}",
                flush=True,
            )
            self.get_logger().info(
                f"Perception ({perception_version}) — port at "
                f"[{port_transform.translation.x:.4f}, "
                f"{port_transform.translation.y:.4f}, "
                f"{port_transform.translation.z:.4f}]"
            )
            result = self._run_with_perception(
                task, get_observation, move_robot, send_feedback,
                port_transform, port_z_axis,
            )
            if self._dagger_collect and self._dagger_steps:
                self._save_dagger_episode()
            if self._ppo_train and self._ppo_trajectory:
                self._save_ppo_trajectory(success=result)
            self._collector.finish_episode(success=result)
            return result

        # DAgger fallback: if perception failed but TF is available, use TF position
        if self._dagger_collect and has_port_tf:
            self.get_logger().warn("DAgger: perception failed, using TF position as fallback")
            try:
                port_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", port_frame, Time()
                )
                port_transform = port_tf.transform
                port_z_axis = self._get_port_z_axis(port_transform)
                result = self._run_with_perception(
                    task, get_observation, move_robot, send_feedback,
                    port_transform, port_z_axis,
                )
                if self._dagger_steps:
                    self._save_dagger_episode()
                if self._ppo_train and self._ppo_trajectory:
                    self._save_ppo_trajectory(success=result)
                self._collector.finish_episode(success=result)
                return result
            except TransformException:
                pass

        print("[SmartInsert] PERCEPTION FAILED — falling back to blind descent", flush=True)
        self.get_logger().warn("Perception failed — using blind-descent fallback")
        result = self._run_blind_descent(
            task, get_observation, move_robot, send_feedback,
        )
        if self._ppo_train and self._ppo_trajectory:
            self._save_ppo_trajectory(success=result)
        self._collector.finish_episode(success=result)
        return result

    # ==================================================================
    #  MODE 1:  TF-based insertion  (ground_truth:=true)
    #
    #  We know the exact position AND orientation of the port, so we can
    #  compute the insertion axis and approach from the correct direction.
    #
    #  Phase 1 — APPROACH:  smoothly fly from current pose to a point
    #            offset along the port's insertion axis (5 s sim-time).
    #  Phase 2 — DESCEND:   move along the insertion axis until the plug
    #            is 15 mm past the port center.
    #  Phase 3 — HOLD:      keep the pose steady while scoring checks.
    # ==================================================================

    def _run_with_tf(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Look up the port's pose in base_link.
        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().error(f"Port TF lookup failed: {ex}")
            return True

        port_transform = port_tf.transform
        pp = port_transform.translation
        port_xyz = np.array([pp.x, pp.y, pp.z])
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"

        # Compute the port's insertion axis (local Z direction in base_link).
        # Positive Z points "into" the port; the plug approaches from -Z.
        port_z = self._get_port_z_axis(port_transform)
        self.get_logger().info(
            f"Port at x={pp.x:.4f} y={pp.y:.4f} z={pp.z:.4f}  "
            f"ins_axis=[{port_z[0]:.3f}, {port_z[1]:.3f}, {port_z[2]:.3f}]"
        )

        def _plug_xyz_from_tf():
            try:
                pl = self._parent_node._tf_buffer.lookup_transform("base_link", plug_frame, Time())
                return np.array([pl.transform.translation.x, pl.transform.translation.y, pl.transform.translation.z])
            except TransformException:
                return None

        # Helper: record one step of training data (no-op when collector is off).
        def _record(obs, target, z_off, phase):
            if not self._collector.enabled:
                return
            try:
                p_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", port_frame, Time()
                ).transform
            except TransformException:
                p_tf = port_transform
            try:
                pl_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", plug_frame, Time()
                ).transform
            except TransformException:
                pl_tf = None
            self._collector.record_step(
                obs=obs,
                expert_target=target,
                port_tf=p_tf,
                plug_tf=pl_tf,
                z_offset=z_off,
                phase=phase,
                insertion_axis=port_z,
            )

        # --- Phase 1: Approach ------------------------------------------------
        send_feedback("Phase 1: approaching port")
        z_offset = 0.20  # start 20 cm away from port along insertion axis
        n_approach = 100  # 100 steps × 0.05 s = 5 s sim-time

        for t in range(n_approach):
            if self._time_remaining() <= 15:
                break
            frac = (t + 1) / n_approach  # ramps from 0.01 → 1.0

            try:
                base_target = self._calc_gripper_pose(
                    port_transform,
                    port_z_axis=port_z,
                    slerp_fraction=frac,
                    position_fraction=frac,
                    z_offset=z_offset,
                    reset_integrator=True,
                )
                obs = get_observation()
                if obs is not None:
                    tcp = obs.controller_state.tcp_pose
                    base_action_world = np.array([
                        base_target.position.x - tcp.position.x,
                        base_target.position.y - tcp.position.y,
                        base_target.position.z - tcp.position.z,
                        0.0, 0.0, 0.0,
                    ])
                    target, _imp = self._apply_residual(
                        base_target, obs, port_xyz, port_z, z_offset,
                        base_action_world=base_action_world,
                    )
                    self._record_dagger_step(obs, base_target, port_xyz, port_z, z_offset, base_action_world)
                    _record(obs, target, z_offset, "approach")
                    pxyz = _plug_xyz_from_tf()
                    if pxyz is not None:
                        self._record_residual_step(obs, target, port_xyz, port_z, z_offset, pxyz)
                else:
                    target = base_target
                self.set_pose_target(move_robot=move_robot, pose=target)
            except TransformException as ex:
                self.get_logger().warn(f"TF during approach: {ex}")
            self.sleep_for(0.05)

        # Settle for 1 second at the approach pose (let vibrations die).
        for _ in range(20):
            try:
                base_target = self._calc_gripper_pose(
                    port_transform,
                    port_z_axis=port_z,
                    z_offset=z_offset,
                    reset_integrator=True,
                )
                obs = get_observation()
                if obs is not None:
                    tcp = obs.controller_state.tcp_pose
                    base_action_world = np.array([
                        base_target.position.x - tcp.position.x,
                        base_target.position.y - tcp.position.y,
                        base_target.position.z - tcp.position.z,
                        0.0, 0.0, 0.0,
                    ])
                    target, _imp = self._apply_residual(
                        base_target, obs, port_xyz, port_z, z_offset,
                        base_action_world=base_action_world,
                    )
                else:
                    target = base_target
                self.set_pose_target(move_robot=move_robot, pose=target)
            except TransformException:
                pass
            self.sleep_for(0.05)

        self.get_logger().info("Approach complete, beginning descent")

        is_sc_task = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()
        descent_limit = -0.045 if is_sc_task else -0.025
        min_seated_depth = -0.035 if is_sc_task else -0.020
        seated = False
        _seated_streak = 0

        for attempt in range(1, MAX_INSERTION_ATTEMPTS + 1):
            if self._time_remaining() <= RETRY_MIN_TIME and attempt > 1:
                self.get_logger().info(f"Not enough time for attempt {attempt}, skipping")
                break

            # --- Phase 2: Descend along insertion axis ------------------------
            send_feedback(f"Phase 2: descent (attempt {attempt}/{MAX_INSERTION_ATTEMPTS})")
            self.get_logger().info(
                f"=== Insertion attempt {attempt}/{MAX_INSERTION_ATTEMPTS} "
                f"z_offset={z_offset:.4f} time_remaining={self._time_remaining():.0f}s ==="
            )
            self._current_attempt = attempt

            # Capture force baselines
            _baseline_obs = get_observation()
            if _baseline_obs is not None:
                _bl, _bz, _ = self._get_force_magnitude(_baseline_obs)
            else:
                _bl, _bz = 0.0, 0.0
            _axial_history: list = []
            _current_phase = PHASE_FREE_SPACE
            _force_deriv = 0.0
            _seated_streak = 0
            if self._phase_hmm is not None:
                self._phase_hmm.reset()

            # PPO retry penalty: immediate negative reward for each retry
            if attempt > 1 and self._ppo_train and self._ppo_trajectory:
                retry_penalty = -2.0
                self._ppo_trajectory[-1]["reward"] += retry_penalty
                self.get_logger().info(f"PPO: retry penalty {retry_penalty} applied")

            while z_offset > descent_limit:
                if self._time_remaining() <= 10:
                    self.get_logger().warn("Time running low, stopping descent")
                    break

                z_offset -= 0.0005

                try:
                    base_target = self._calc_gripper_pose(
                        port_transform,
                        port_z_axis=port_z,
                        z_offset=z_offset,
                    )
                    obs = get_observation()
                    if obs is not None:
                        _lat_f, _z_f, _ = self._get_force_magnitude(obs)
                        _axial_history.append(_z_f)
                        if len(_axial_history) > 2:
                            _force_deriv = _axial_history[-1] - _axial_history[-2]

                        if self._phase_hmm is not None:
                            _f_var = float(np.var(_axial_history[-10:])) if len(_axial_history) >= 2 else 0.0
                            _hmm_obs = np.array([
                                abs(_lat_f - _bl), abs(_z_f - _bz),
                                z_offset, _force_deriv, _f_var,
                            ])
                            _current_phase = self._phase_hmm.update(_hmm_obs, min_seated_depth=min_seated_depth)
                            _seated_streak = self._phase_hmm.seated_streak
                        else:
                            _raw_phase = self._classify_contact_state(
                                _lat_f, _z_f, _bl, _bz,
                                z_offset, _axial_history,
                                min_seated_depth=min_seated_depth,
                            )
                            if _raw_phase == PHASE_SEATED:
                                _seated_streak += 1
                                _current_phase = PHASE_SEATED if _seated_streak >= 5 else PHASE_INSERTION
                            else:
                                _seated_streak = 0
                                _current_phase = _raw_phase

                        tcp = obs.controller_state.tcp_pose
                        base_action_world = np.array([
                            base_target.position.x - tcp.position.x,
                            base_target.position.y - tcp.position.y,
                            base_target.position.z - tcp.position.z,
                            0.0, 0.0, 0.0,
                        ])
                        target, _imp = self._apply_residual(
                            base_target, obs, port_xyz, port_z, z_offset,
                            base_action_world=base_action_world,
                            phase=_current_phase,
                            force_deriv_axial=_force_deriv,
                        )
                        self._record_dagger_step(obs, base_target, port_xyz, port_z, z_offset, base_action_world)
                        _record(obs, target, z_offset, "descent")
                        pxyz = _plug_xyz_from_tf()
                        if pxyz is not None:
                            self._record_residual_step(obs, target, port_xyz, port_z, z_offset, pxyz)
                    else:
                        target = base_target
                    self.set_pose_target(move_robot=move_robot, pose=target)
                except TransformException as ex:
                    self.get_logger().warn(f"TF during descent: {ex}")
                self.sleep_for(0.05)

                if z_offset < 0.05 and int(z_offset * 10000) % 50 == 0:
                    self.get_logger().info(
                        f"z_offset={z_offset:.4f} phase={_current_phase} seated_streak={_seated_streak}"
                    )

            # Check if insertion succeeded or is progressing well
            if _current_phase >= PHASE_SEATED:
                seated = True
                self.get_logger().info(
                    f">>> SEATED on attempt {attempt}! z_offset={z_offset:.4f}"
                )
                break

            if _current_phase >= PHASE_INSERTION:
                # Plug is entering the port — hold here, don't retract
                self.get_logger().info(
                    f"Attempt {attempt}: INSERTION phase reached (z_offset={z_offset:.4f}) "
                    f"— holding position (no retract needed)"
                )
                break

            # Clearly failed (FREE_SPACE / NEAR_CONTACT / ALIGNMENT) — retry
            if attempt < MAX_INSERTION_ATTEMPTS and self._time_remaining() > RETRY_MIN_TIME:
                self.get_logger().info(
                    f"Attempt {attempt} did NOT reach insertion (phase={_current_phase}). "
                    f"Retracting for retry..."
                )
                send_feedback(f"Retracting for attempt {attempt + 1}")

                retract_target_z = RETRACT_HEIGHT
                retract_steps = int((retract_target_z - z_offset) / 0.001)
                for _ in range(min(retract_steps, 200)):
                    z_offset += 0.001
                    try:
                        target = self._calc_gripper_pose(
                            port_transform,
                            port_z_axis=port_z,
                            z_offset=z_offset,
                        )
                        self.set_pose_target(move_robot=move_robot, pose=target)
                    except TransformException:
                        pass
                    self.sleep_for(0.03)

                for _ in range(RETRY_SETTLE_STEPS):
                    try:
                        target = self._calc_gripper_pose(
                            port_transform,
                            port_z_axis=port_z,
                            z_offset=z_offset,
                        )
                        self.set_pose_target(move_robot=move_robot, pose=target)
                    except TransformException:
                        pass
                    self.sleep_for(0.05)

                self.get_logger().info(
                    f"Retracted to z_offset={z_offset:.4f}, ready for attempt {attempt + 1}"
                )
                self._tip_error_integrator = np.array([0.0, 0.0, 0.0])
            else:
                self.get_logger().info(
                    f"Attempt {attempt} failed (phase={_current_phase}) — no more retries "
                    f"(time={self._time_remaining():.0f}s)"
                )

        # --- Phase 3: Hold for scoring verification --------------------------
        send_feedback("Phase 3: holding for verification")
        self.get_logger().info(
            f"Hold at z_offset={z_offset:.4f} seated={seated} attempts_used={attempt}"
        )
        hold_secs = min(5.0, max(0.0, self._time_remaining() - 2.0))
        hold_steps = int(hold_secs / 0.05)

        for _ in range(hold_steps):
            try:
                target = self._calc_gripper_pose(
                    port_transform,
                    port_z_axis=port_z,
                    z_offset=z_offset,
                )
                self.set_pose_target(move_robot=move_robot, pose=target)
                obs = get_observation()
                if obs is not None:
                    _record(obs, target, z_offset, "hold")
                    pxyz = _plug_xyz_from_tf()
                    if pxyz is not None:
                        self._record_residual_step(obs, target, port_xyz, port_z, z_offset, pxyz)
            except TransformException:
                pass
            self.sleep_for(0.05)

        self.get_logger().info(
            f"SmartInsert complete (TF mode) — seated={seated}, attempts={attempt}"
        )
        return True

    # ==================================================================
    #  PERCEPTION MODE: port from camera (V1/V2), no plug TF
    #  Same approach/descent/hold as TF mode but gripper pose from obs only.
    # ==================================================================

    def _record_residual_step(
        self,
        obs: Observation,
        expert_target: Pose,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        z_offset: float,
        plug_xyz: np.ndarray,
    ) -> None:
        """Record (state_20dim, residual_6dim) for YD-RRL training (TF mode only)."""
        if not self._collect_residual:
            return
        try:
            import sys
            from pathlib import Path
            rl = Path.home() / "rl"
            if str(rl) not in sys.path:
                sys.path.insert(0, str(rl))
            from training.frame_decomposer import yaw_from_insertion_axis, yaw_rotation_matrix
            from training.residual_mlp import build_yd_rrl_state
        except ImportError:
            return
        base_target = self._calc_gripper_pose_from_observation(obs, port_xyz, port_z_axis, z_offset)
        residual_pos_world = np.array([
            expert_target.position.x - base_target.position.x,
            expert_target.position.y - base_target.position.y,
            expert_target.position.z - base_target.position.z,
        ])
        yaw = yaw_from_insertion_axis(port_z_axis)
        R = yaw_rotation_matrix(yaw)
        residual_local_pos = R @ residual_pos_world
        residual_local = np.concatenate([residual_local_pos, np.zeros(3)]).astype(np.float32)
        residual_local = np.clip(residual_local, -0.001, 0.001)  # ±1mm
        w = obs.wrist_wrench.wrench
        F = np.array([w.force.x, w.force.y, w.force.z])
        τ = np.array([w.torque.x, w.torque.y, w.torque.z])
        pose_error_world = np.concatenate([port_xyz - plug_xyz, np.zeros(3)])
        insertion_progress = 1.0 - (z_offset + 0.015) / 0.215
        insertion_progress = np.clip(insertion_progress, 0.0, 1.0)
        contact = 1.0 if np.linalg.norm(F) > 2.0 else 0.0
        connector_sfp = self._task and "sfp" in str(self._task.port_type or "").lower()
        time_rem = max(0.0, self._time_remaining() / max(1.0, self._time_budget))
        state = build_yd_rrl_state(
            F, τ, pose_error_world, yaw,
            insertion_progress, contact, connector_sfp, time_rem,
            cable_tension_est=None,
        )
        self._residual_steps.append((state, residual_local))

    def _save_residual_episode(self) -> None:
        """Write collected (state, residual) steps to ~/rl/residual_data/episode_XXXX.npz."""
        from pathlib import Path
        self._residual_data_dir = Path(self._residual_data_dir)
        self._residual_data_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self._residual_data_dir.glob("episode_*.npz"))
        self._residual_episode_num = max(
            (int(p.stem.split("_")[1]) for p in existing),
            default=-1,
        ) + 1
        path = self._residual_data_dir / f"episode_{self._residual_episode_num:04d}.npz"
        states = np.array([s for s, _ in self._residual_steps], dtype=np.float32)
        residuals = np.array([r for _, r in self._residual_steps], dtype=np.float32)
        np.savez_compressed(path, states=states, residuals=residuals)
        self.get_logger().info(
            f"Residual data: saved {len(self._residual_steps)} steps → {path}"
        )
        self._residual_steps = []

    # ------------------------------------------------------------------
    #  DAgger data collection helpers
    # ------------------------------------------------------------------

    def _record_dagger_step(
        self,
        obs: Observation,
        base_target: Pose,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        z_offset: float,
        base_action_world: np.ndarray,
    ) -> None:
        """Record one DAgger step: (state_26D, residual_24D) with TF expert labels."""
        if not self._dagger_collect or not self._dagger_port_frame:
            return
        try:
            import sys
            from pathlib import Path
            rl = Path.home() / "rl"
            if str(rl) not in sys.path:
                sys.path.insert(0, str(rl))
            from training.frame_decomposer import (
                yaw_from_insertion_axis, yaw_rotation_matrix,
                world_to_port_local_force_torque,
            )
            from training.residual_mlp import build_yd_rrl_state, POS_BOUND
        except ImportError:
            return

        # Look up TF expert target
        try:
            port_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link", self._dagger_port_frame, Time()
            )
            expert_target = self._calc_gripper_pose(
                port_tf.transform,
                port_z_axis=port_z_axis,
                z_offset=z_offset,
            )
        except TransformException:
            return

        yaw = yaw_from_insertion_axis(port_z_axis)
        R = yaw_rotation_matrix(yaw)

        # Pose residual in port-local frame (expert - base)
        pose_res_world = np.array([
            expert_target.position.x - base_target.position.x,
            expert_target.position.y - base_target.position.y,
            expert_target.position.z - base_target.position.z,
        ])
        pose_res_local = np.clip(R @ pose_res_world, -POS_BOUND, POS_BOUND).astype(np.float32)
        orient_res_local = np.zeros(3, dtype=np.float32)

        # Impedance heuristic labels from force/velocity
        w = obs.wrist_wrench.wrench
        F = np.array([w.force.x, w.force.y, w.force.z])
        tau = np.array([w.torque.x, w.torque.y, w.torque.z])
        F_local, tau_local = world_to_port_local_force_torque(F[:3], tau[:3], yaw)
        force_mag = np.linalg.norm(F)

        delta_K, delta_D, delta_F = self._compute_impedance_heuristic(
            obs, F_local, tau_local, force_mag,
        )

        residual_24d = np.concatenate([
            pose_res_local, orient_res_local,
            delta_K, delta_D, delta_F,
        ]).astype(np.float32)

        # Build 26D state
        tcp = obs.controller_state.tcp_pose
        tcp_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        pose_error_world = np.concatenate([port_xyz - tcp_xyz, np.zeros(3)])
        insertion_progress = np.clip(1.0 - (z_offset + 0.015) / 0.215, 0.0, 1.0)
        contact_flag = 1.0 if force_mag > 2.0 else 0.0
        connector_sfp = self._task and "sfp" in str(self._task.port_type or "").lower()
        time_rem = max(0.0, self._time_remaining() / max(1.0, self._time_budget))

        ba_world = np.asarray(base_action_world).ravel()
        base_action_local = np.zeros(6, dtype=np.float32)
        if len(ba_world) >= 3:
            base_action_local[:3] = R @ ba_world[:3]
        if len(ba_world) >= 6:
            base_action_local[3:6] = R @ ba_world[3:6]

        state = build_yd_rrl_state(
            F, tau, pose_error_world, yaw,
            insertion_progress, contact_flag, connector_sfp, time_rem,
            base_action_local=base_action_local,
            cable_tension_est=None,
        )

        self._dagger_steps.append({
            'state': state,
            'residual': residual_24d,
            'base_action': base_action_local.copy(),
        })

    @staticmethod
    def _compute_impedance_heuristic(
        obs: Observation,
        F_local: np.ndarray,
        tau_local: np.ndarray,
        force_mag: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute heuristic impedance labels (ΔK, ΔD, ΔF) from sensor data.

        Returns clipped arrays each of shape (6,).
        """
        # ΔK: soften when contact force is high
        gain_k = 2.0
        dk = -gain_k * max(0.0, force_mag - 5.0)
        delta_K = np.full(6, dk, dtype=np.float32)
        delta_K = np.clip(delta_K, -30.0, 30.0)

        # ΔD: increase damping near contact proportional to velocity
        contact = 1.0 if force_mag > 2.0 else 0.0
        try:
            v = obs.controller_state.tcp_linear_velocity
            vel_mag = np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        except Exception:
            vel_mag = 0.0
        gain_d = 5.0
        dd = gain_d * vel_mag * contact
        delta_D = np.full(6, dd, dtype=np.float32)
        delta_D = np.clip(delta_D, -20.0, 20.0)

        # ΔF: compensate undesired lateral forces / torques
        gain_f, gain_t = 0.3, 0.1
        delta_F = np.zeros(6, dtype=np.float32)
        delta_F[:3] = np.clip(-gain_f * F_local[:3], -5.0, 5.0)
        delta_F[3:6] = np.clip(-gain_t * tau_local[:3], -1.0, 1.0)
        return delta_K, delta_D, delta_F

    def _save_dagger_episode(self) -> None:
        """Write collected DAgger (state, residual) steps to disk."""
        from pathlib import Path
        self._dagger_data_dir = Path(self._dagger_data_dir)
        self._dagger_data_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self._dagger_data_dir.glob("episode_*.npz"))
        ep_num = max(
            (int(p.stem.split("_")[1]) for p in existing), default=-1,
        ) + 1
        path = self._dagger_data_dir / f"episode_{ep_num:04d}.npz"
        states = np.array([s['state'] for s in self._dagger_steps], dtype=np.float32)
        residuals = np.array([s['residual'] for s in self._dagger_steps], dtype=np.float32)
        base_actions = np.array([s['base_action'] for s in self._dagger_steps], dtype=np.float32)
        np.savez_compressed(
            path, states=states, residuals=residuals, base_actions=base_actions,
        )
        self.get_logger().info(
            f"DAgger round {self._dagger_round}: saved {len(self._dagger_steps)} steps → {path}"
        )
        self._dagger_steps = []

    def _set_pose_target_soft(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        impedance: Optional[dict] = None,
    ) -> None:
        """Send pose command with impedance parameters.

        If impedance dict is provided (from residual MLP), use its learned
        stiffness/damping/wrench. Otherwise use the base soft-Z defaults.
        """
        from std_msgs.msg import Header
        from geometry_msgs.msg import Vector3, Wrench

        if impedance is not None:
            K = impedance["stiffness"]
            D = impedance["damping"]
            F_ff = impedance["feedforward_wrench"]
        else:
            K = np.array([90.0, 90.0, 40.0, 40.0, 40.0, 40.0])
            D = np.array([50.0, 50.0, 30.0, 20.0, 20.0, 20.0])
            F_ff = np.zeros(6)

        motion_update = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(K).flatten(),
            target_damping=np.diag(D).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=float(F_ff[0]), y=float(F_ff[1]), z=float(F_ff[2])),
                torque=Vector3(x=float(F_ff[3]), y=float(F_ff[4]), z=float(F_ff[5])),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:
            self.get_logger().info(f"move_robot exception: {ex}")

    @staticmethod
    def _clamp_to_workspace(pos: np.ndarray) -> np.ndarray:
        """Clamp a position to the safe workspace bounds."""
        return np.clip(pos, WORKSPACE_MIN, WORKSPACE_MAX)

    @staticmethod
    def _get_force_magnitude(obs) -> tuple:
        """Return (lateral_force, z_force, total_force) from observation."""
        try:
            fx = obs.wrist_wrench.wrench.force.x
            fy = obs.wrist_wrench.wrench.force.y
            fz = obs.wrist_wrench.wrench.force.z
            lateral = np.sqrt(fx * fx + fy * fy)
            return lateral, abs(fz), np.sqrt(fx*fx + fy*fy + fz*fz)
        except Exception:
            return 0.0, 0.0, 0.0

    def _get_plug_offset(self, settled: bool = True) -> np.ndarray:
        """Return calibrated TCP-to-plug offset based on connector type.

        Args:
            settled: If True, return the offset for the descent/hold phase
                     (cable taut, plug directly below gripper).  If False,
                     return the approach-phase offset (plug hanging loose).
        """
        is_sc = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()
        if is_sc:
            return (PLUG_OFFSET_SC_SETTLED if settled else PLUG_OFFSET_SC_APPROACH).copy()
        return (PLUG_OFFSET_SFP_SETTLED if settled else PLUG_OFFSET_SFP_APPROACH).copy()

    def _detect_with_multi_obs(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        n_obs: int = 3,
        delay: float = 0.15,
        min_cams: int = 2,
    ):
        """Take n_obs observations and average V2 predictions (uncertainty-weighted).

        Returns (port_transform, port_z_axis) or None.
        Only includes detections that meet the min_cams threshold.
        """
        import time as _time
        from smart_insert.perception.v2_learned import detect_port_v2
        predictions = []

        for i in range(n_obs):
            print(f"[SmartInsert] _detect_with_multi_obs obs {i+1}/{n_obs}", flush=True)
            obs = get_observation()
            if obs is None:
                print(f"[SmartInsert] obs {i+1} is None, sleeping {delay}s", flush=True)
                _time.sleep(delay)
                continue
            print(f"[SmartInsert] obs {i+1}: calling detect_port_v2...", flush=True)
            result = detect_port_v2(
                obs, task, self._parent_node._tf_buffer, self.get_logger(),
                min_cams=min_cams,
            )
            if result is not None:
                print(f"[SmartInsert] obs {i+1}: V2 returned a detection ({result[2]}/3 cams)", flush=True)
                predictions.append(result)
            else:
                print(f"[SmartInsert] obs {i+1}: V2 returned None", flush=True)
            if i < n_obs - 1:
                _time.sleep(max(delay, 0.5))

        if not predictions:
            self.get_logger().warn("V2: all observations failed, no predictions")
            return None

        if len(predictions) == 1:
            return (predictions[0][0], predictions[0][1])

        positions = np.array([
            [r[0].translation.x, r[0].translation.y, r[0].translation.z]
            for r in predictions
        ])
        avg_pos = positions.mean(axis=0)
        spread_mm = np.std(positions, axis=0).max() * 1000
        avg_cams = np.mean([r[2] for r in predictions])
        self.get_logger().info(
            f"V2: averaged {len(predictions)} detections, "
            f"spread={spread_mm:.1f}mm, avg_cams={avg_cams:.1f}"
        )

        port_transform = Transform()
        port_transform.translation.x = float(avg_pos[0])
        port_transform.translation.y = float(avg_pos[1])
        port_transform.translation.z = float(avg_pos[2])
        port_transform.rotation.w = 1.0
        port_transform.rotation.x = 0.0
        port_transform.rotation.y = 0.0
        port_transform.rotation.z = 0.0

        port_z_axis = predictions[0][1]
        return (port_transform, port_z_axis)

    def _detect_single_obs(
        self,
        obs: Observation,
        task: Task,
        min_cams: int = 3,
    ):
        """Run V2 detection on a single observation (fast, no multi-obs averaging).

        Returns (port_xyz_array, axis_array, n_confident_cams) or None.
        min_cams: minimum confident cameras required (default 3 for high quality).
        """
        from smart_insert.perception.v2_learned import detect_port_v2
        result = detect_port_v2(
            obs, task, self._parent_node._tf_buffer, self.get_logger(),
            min_cams=min_cams,
        )
        if result is None:
            return None
        port_tf, axis, n_cams = result
        xyz = np.array([port_tf.translation.x, port_tf.translation.y, port_tf.translation.z])
        return (xyz, axis, n_cams)

    def _update_port_estimate(
        self,
        old_xyz: np.ndarray,
        new_xyz: np.ndarray,
        alpha: float,
        max_shift: float = 0.015,
    ) -> np.ndarray:
        """EMA update of port estimate, clamping max single-step shift.

        alpha: blend factor for the new detection (0 = keep old, 1 = use new)
        max_shift: maximum allowed position change per update (m)
        """
        delta = new_xyz - old_xyz
        shift = np.linalg.norm(delta)
        if shift > max_shift:
            delta = delta * (max_shift / shift)
        return old_xyz + alpha * delta

    def _load_residual_mlp(self):
        """Load residual policy: PPO (primary) + DAgger (fallback), or DAgger-only.

        Priority order:
          1. HybridResidualPolicy (PPO + DAgger) — if PPO checkpoint exists
          2. ResidualMLP (DAgger only) — fallback
          3. Zero residual — if nothing found
        """
        import sys
        from pathlib import Path
        rl = Path.home() / "rl"
        if not rl.exists():
            return None
        if str(rl) not in sys.path:
            sys.path.insert(0, str(rl))

        ckpt_env = __import__("os").environ.get("AIC_RESIDUAL_CHECKPOINT", "")

        # Try hybrid PPO+DAgger first
        try:
            ppo_dir = rl / "yd_rrl_checkpoints" / "ppo_resip"
            dagger_dir = Path(ckpt_env) if ckpt_env else rl / "yd_rrl_checkpoints" / "dagger_r3"

            if (ppo_dir / "ppo_actor.pt").exists():
                from training.ppo_residual import HybridResidualPolicy
                hybrid = HybridResidualPolicy(
                    ppo_checkpoint=str(ppo_dir),
                    dagger_checkpoint=str(dagger_dir) if dagger_dir.exists() else None,
                    entropy_threshold=2.0,
                    blend_alpha=0.7,
                )
                self.get_logger().info(
                    f"Hybrid ResiP loaded: PPO={ppo_dir}, DAgger fallback={'yes' if dagger_dir.exists() else 'no'}"
                )
                return hybrid
        except Exception as e:
            self.get_logger().info(f"PPO hybrid not available ({e}), trying DAgger-only")

        # Fall back to DAgger-only
        try:
            from training.residual_mlp import ResidualMLP
            if not ckpt_env:
                default = rl / "yd_rrl_checkpoints" / "dagger_r3"
                if (default / "residual_mlp.pt").exists():
                    ckpt_env = str(default)
            return ResidualMLP(checkpoint_path=ckpt_env or None)
        except Exception as e:
            self.get_logger().warn(f"Residual MLP load failed: {e}")
            return None

    def _apply_residual(
        self,
        target: Pose,
        obs: Observation,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        z_offset: float,
        base_action_world: Optional[np.ndarray] = None,
        phase: int = 0,
        force_deriv_axial: float = 0.0,
    ) -> tuple[Pose, Optional[dict]]:
        """Apply impedance residual correction (Impedance-Aware ResiP).

        Returns (corrected_pose, impedance_dict_or_None).
        impedance_dict has 'stiffness'(6), 'damping'(6), 'feedforward_wrench'(6)
        when the residual is active, or None to use defaults.
        """
        if self._residual_mlp is None and not self._ppo_train:
            return target, None
        try:
            import sys
            from pathlib import Path
            rl = Path.home() / "rl"
            if str(rl) not in sys.path:
                sys.path.insert(0, str(rl))
            from training.frame_decomposer import yaw_from_insertion_axis, port_local_to_world_delta, yaw_rotation_matrix
            from training.residual_mlp import build_yd_rrl_state, compute_impedance, build_contact_features
        except ImportError as ie:
            if not getattr(self, '_residual_import_warned', False):
                self.get_logger().error(f"_apply_residual ImportError: {ie}")
                self._residual_import_warned = True
            return target, None
        w = obs.wrist_wrench.wrench
        F = np.array([w.force.x, w.force.y, w.force.z])
        τ = np.array([w.torque.x, w.torque.y, w.torque.z])
        tcp = obs.controller_state.tcp_pose
        tcp_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
        pose_error_world = port_xyz - tcp_xyz
        pose_error_world = np.concatenate([pose_error_world, np.zeros(3)])
        yaw = yaw_from_insertion_axis(port_z_axis)

        # Compute lateral force for contact features
        F_along = np.dot(F, port_z_axis) * port_z_axis
        F_lateral = F - F_along
        contact_feat = build_contact_features(
            phase=phase,
            force_deriv_axial=force_deriv_axial,
            lateral_force_world=F_lateral,
            port_z_axis=port_z_axis,
        )

        # Compute base action in port-local frame (ResiP: state includes base action)
        base_action_local = np.zeros(6, dtype=np.float32)
        if base_action_world is not None:
            R = yaw_rotation_matrix(yaw)
            ba = np.asarray(base_action_world).ravel()
            if len(ba) >= 3:
                base_action_local[:3] = R @ ba[:3]
            if len(ba) >= 6:
                base_action_local[3:6] = R @ ba[3:6]

        insertion_progress = np.clip(1.0 - (z_offset + 0.015) / 0.215, 0.0, 1.0)
        contact = 1.0 if np.linalg.norm(F) > 2.0 else 0.0
        connector_sfp = self._task and "sfp" in str(self._task.port_type or "").lower()
        time_rem = max(0.0, self._time_remaining() / max(1.0, self._time_budget))
        state = build_yd_rrl_state(
            F, τ, pose_error_world, yaw,
            insertion_progress, contact, connector_sfp, time_rem,
            base_action_local=base_action_local,
            contact_features=contact_feat,
        )

        # PPO training mode: stochastic policy + trajectory recording
        if not getattr(self, '_residual_debug_logged', False):
            self.get_logger().info(
                f"_apply_residual: ppo_train={self._ppo_train}, "
                f"ppo_actor={self._ppo_actor is not None}, "
                f"residual_mlp={self._residual_mlp is not None}"
            )
            self._residual_debug_logged = True
        if self._ppo_train and self._ppo_actor is not None:
            residual_action = self._ppo_apply_stochastic(
                state, z_offset, obs=obs,
                port_xyz=port_xyz, port_z_axis=port_z_axis, phase=phase,
            )
        else:
            residual_action = self._residual_mlp.forward(state) * RESIDUAL_ALPHA

            # Log source when using hybrid policy (PPO vs DAgger fallback)
            if hasattr(self._residual_mlp, 'last_source'):
                src = self._residual_mlp.last_source
                if src == "blend" and not hasattr(self, '_last_blend_log_step'):
                    self._last_blend_log_step = 0
                if src == "blend":
                    cnt = getattr(self, '_blend_count', 0) + 1
                    self._blend_count = cnt
                    if cnt % 20 == 1:
                        ent = getattr(self._residual_mlp, 'last_entropy', 0)
                        self.get_logger().info(
                            f"ResiP: DAgger fallback active (entropy={ent:.2f}, count={cnt})"
                        )
                elif src == "ppo":
                    self._blend_count = 0

        # Pose correction (first 6 dims)
        residual_world = port_local_to_world_delta(residual_action[:6], yaw)
        from geometry_msgs.msg import Point
        corrected_pose = Pose(
            position=Point(
                x=target.position.x + float(residual_world[0]),
                y=target.position.y + float(residual_world[1]),
                z=target.position.z + float(residual_world[2]),
            ),
            orientation=target.orientation,
        )

        # Impedance modulation (dims 6-23, only if V2 model)
        impedance = None
        if len(residual_action) > 6:
            impedance = compute_impedance(residual_action)

        return corrected_pose, impedance

    # ------------------------------------------------------------------
    #  PPO Training: stochastic action + trajectory recording
    # ------------------------------------------------------------------

    def _ppo_apply_stochastic(
        self,
        state: np.ndarray,
        z_offset: float,
        obs=None,
        port_xyz: np.ndarray = None,
        port_z_axis: np.ndarray = None,
        phase: int = 0,
    ) -> np.ndarray:
        """Sample action from PPO Gaussian policy and record trajectory step."""
        import torch
        from training.residual_mlp import clip_action, RESIDUAL_ALPHA

        if not getattr(self, '_ppo_stochastic_logged', False):
            self.get_logger().info(f"PPO stochastic: first call, z_offset={z_offset:.4f}")
            self._ppo_stochastic_logged = True

        with torch.no_grad():
            s_t = torch.from_numpy(state).float().unsqueeze(0)
            action, log_prob, entropy = self._ppo_actor.get_action(s_t, deterministic=False)
            value = self._ppo_critic(s_t)

        action_np = clip_action(action.squeeze(0).numpy())

        reward = self._ppo_compute_reward(
            z_offset, obs=obs, port_xyz=port_xyz,
            port_z_axis=port_z_axis, phase=phase,
        )

        self._ppo_trajectory.append({
            "state": state.copy(),
            "action": action_np.copy(),
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": reward,
            "z_offset": z_offset,
        })

        return action_np * RESIDUAL_ALPHA

    def _ppo_compute_reward(
        self,
        z_offset: float,
        obs=None,
        port_xyz: np.ndarray = None,
        port_z_axis: np.ndarray = None,
        phase: int = 0,
    ) -> float:
        """Dense reward combining distance progress + force signals + phase bonuses.

        For SFP: uses TF plug-port distance (smooth, works well).
        For SC: uses TCP-to-port distance (plug TF frame doesn't track approach).
        Both: add force-based rewards that transfer to no-TF evaluation.

        Attempt-aware: higher reward for first-attempt success, diminishing
        with retries to encourage the policy to get it right the first time.
        """
        from tf2_ros import TransformException

        reward = -0.01  # time penalty
        attempt = getattr(self, '_current_attempt', 1)

        is_sc = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()

        # --- Distance-based reward ---
        dist = None
        try:
            if is_sc and obs is not None and port_xyz is not None:
                tcp = obs.controller_state.tcp_pose
                tcp_pos = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
                plug_off = self._get_plug_offset(settled=(z_offset < 0.05))
                est_plug_pos = tcp_pos + plug_off
                dist = float(np.linalg.norm(port_xyz - est_plug_pos))
            else:
                port_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", self._ppo_port_frame, Time(),
                )
                plug_tf = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", self._ppo_plug_frame, Time(),
                )
                pp = port_tf.transform.translation
                pl = plug_tf.transform.translation
                port_pos = np.array([pp.x, pp.y, pp.z])
                plug_pos = np.array([pl.x, pl.y, pl.z])
                dist = float(np.linalg.norm(port_pos - plug_pos))
        except TransformException:
            pass

        if dist is not None:
            if self._ppo_prev_plug_port_dist > 0:
                progress = self._ppo_prev_plug_port_dist - dist
                reward += 5.0 * progress
            self._ppo_prev_plug_port_dist = dist

            if dist < 0.02:
                reward += 0.1 * (0.02 - dist) / 0.02

            if is_sc:
                if z_offset < -0.01 and (dist < 0.025 or phase >= PHASE_INSERTION):
                    reward += 10.0
            else:
                if z_offset < -0.01 and dist < 0.01:
                    reward += 10.0

        # --- Force-based reward (works without TF at eval) ---
        if obs is not None and port_z_axis is not None:
            try:
                w = obs.wrist_wrench.wrench
                F = np.array([w.force.x, w.force.y, w.force.z])
                F_axial = abs(float(np.dot(F, port_z_axis)))
                F_lateral = float(np.linalg.norm(F - np.dot(F, port_z_axis) * port_z_axis))

                if z_offset < 0.02 and F_axial > 3.0:
                    reward += 0.05 * min(F_axial / 10.0, 1.0)

                if F_lateral > FUNNEL_CONTACT_THRESHOLD:
                    reward -= 0.02 * min(F_lateral / 10.0, 1.0)
            except Exception:
                pass

        # --- Phase transition bonuses (attempt-scaled) ---
        # First attempt gets full bonus; retries get diminishing rewards
        attempt_multiplier = 1.0 / attempt  # 1.0, 0.5, 0.33...
        if phase == PHASE_INSERTION:
            reward += 0.5 * attempt_multiplier
        elif phase == PHASE_SEATED:
            # Seated bonus: 10.0 on attempt 1, 5.0 on attempt 2, 3.3 on attempt 3
            reward += 10.0 * attempt_multiplier

        return reward

    def _save_ppo_trajectory(self, success: bool = False):
        """Save collected PPO trajectory to disk for batch training."""
        if not self._ppo_trajectory:
            return

        states = np.array([s["state"] for s in self._ppo_trajectory], dtype=np.float32)
        actions = np.array([s["action"] for s in self._ppo_trajectory], dtype=np.float32)
        log_probs = np.array([s["log_prob"] for s in self._ppo_trajectory], dtype=np.float32)
        values = np.array([s["value"] for s in self._ppo_trajectory], dtype=np.float32)
        rewards = np.array([s["reward"] for s in self._ppo_trajectory], dtype=np.float32)
        z_offsets = np.array([s["z_offset"] for s in self._ppo_trajectory], dtype=np.float32)

        # Terminal insertion bonus
        if success and z_offsets[-1] < 0.0:
            rewards[-1] += 20.0

        ep_path = self._ppo_data_dir / f"episode_{self._ppo_episode_num:04d}.npz"
        np.savez_compressed(
            ep_path,
            states=states,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            z_offsets=z_offsets,
            success=np.array([1.0 if success else 0.0]),
        )

        total_reward = float(rewards.sum())
        self.get_logger().info(
            f"PPO trajectory saved: {ep_path.name} "
            f"({len(self._ppo_trajectory)} steps, reward={total_reward:.2f}, "
            f"success={success})"
        )
        self._ppo_episode_num += 1
        self._ppo_trajectory = []

    def _calc_gripper_pose_from_observation(
        self,
        obs: Observation,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        z_offset: float,
        position_fraction: float = 1.0,
        align_orientation: bool = False,
        slerp_fraction: float = 1.0,
    ) -> Pose:
        """Compute gripper target from estimated port and current TCP (no plug TF).

        During approach (position_fraction < 1) we interpolate between the
        approach offset (plug hanging loose) and the settled offset (cable
        taut) so the transition is smooth.  During descent/hold
        (position_fraction == 1.0) we use the settled offset exclusively.
        """
        tcp = obs.controller_state.tcp_pose
        gripper_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z])

        offset_approach = self._get_plug_offset(settled=False)
        offset_settled = self._get_plug_offset(settled=True)
        plug_offset = offset_approach + position_fraction * (offset_settled - offset_approach)

        plug_tip_target = port_xyz - z_offset * port_z_axis
        gripper_target = plug_tip_target + plug_offset

        # Wrist-clearance offset for SC horizontal insertion: lower the gripper
        # during approach/early descent so the wrist link clears the enclosure,
        # then ramp to zero for the final insertion.
        is_sc = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()
        if is_sc and abs(port_z_axis[2]) < 0.5:
            clearance_ramp = min(1.0, max(0.0, z_offset / 0.05))
            gripper_target[2] += SC_WRIST_CLEARANCE_Z * clearance_ramp

        gripper_target = self._clamp_to_workspace(gripper_target)

        blend = gripper_xyz + position_fraction * (gripper_target - gripper_xyz)
        blend = self._clamp_to_workspace(blend)

        orientation = tcp.orientation
        if align_orientation and slerp_fraction > 0:
            sc_tilt = 0.0
            if is_sc and abs(port_z_axis[2]) < 0.5:
                sc_tilt = SC_APPROACH_TILT_DEG * clearance_ramp
            orientation = self._compute_insertion_orientation(
                tcp.orientation, port_z_axis, slerp_fraction,
                tilt_down_deg=sc_tilt,
            )

        return Pose(
            position=Point(x=float(blend[0]), y=float(blend[1]), z=float(blend[2])),
            orientation=orientation,
        )

    def _compute_insertion_orientation(
        self,
        current_orientation: Quaternion,
        port_z_axis: np.ndarray,
        slerp_fraction: float = 1.0,
        tilt_down_deg: float = 0.0,
    ) -> Quaternion:
        """Compute gripper orientation aligned to the insertion axis.

        For a downward insertion axis [0,0,-1] the gripper should point
        its tool straight down. For other axes we compute the rotation that
        aligns the gripper's Z with the port_z_axis.

        tilt_down_deg: additional downward pitch (degrees) to angle the wrist
        away from obstacles above the TCP (used for SC horizontal insertion).
        """
        from scipy.spatial.transform import Rotation as R

        q_current = (
            current_orientation.w,
            current_orientation.x,
            current_orientation.y,
            current_orientation.z,
        )

        approach_dir = port_z_axis.copy()

        if tilt_down_deg > 0.1 and abs(approach_dir[2]) < 0.5:
            tilt_rad = np.radians(tilt_down_deg)
            horiz = np.array([approach_dir[0], approach_dir[1], 0.0])
            h_norm = np.linalg.norm(horiz)
            if h_norm > 1e-6:
                perp = np.cross(horiz / h_norm, np.array([0.0, 0.0, 1.0]))
                approach_dir = R.from_rotvec(tilt_rad * perp).apply(approach_dir)
                approach_dir /= np.linalg.norm(approach_dir)

        # Build target rotation matrix with Z = approach_dir
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(approach_dir, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, approach_dir)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            return current_orientation
        x_axis /= x_norm
        y_axis = np.cross(approach_dir, x_axis)

        rot_mat = np.column_stack([x_axis, y_axis, approach_dir])
        try:
            q_target_scipy = R.from_matrix(rot_mat).as_quat()  # (x,y,z,w)
            q_target = (q_target_scipy[3], q_target_scipy[0],
                        q_target_scipy[1], q_target_scipy[2])
        except Exception:
            return current_orientation

        # Check angular distance: if target requires > 90° rotation from current,
        # the computed frame may be ambiguous (multiple valid orientations exist
        # for a given Z axis). In that case, resolve the ambiguity by choosing the
        # frame closest to current via a 180° flip of x/y axes.
        dot = abs(
            q_current[0] * q_target[0] + q_current[1] * q_target[1]
            + q_current[2] * q_target[2] + q_current[3] * q_target[3]
        )
        if dot < 0.707:  # > 90° rotation needed
            # Flip x and y axes (rotate 180° around z) to get the alternative frame
            rot_mat_alt = np.column_stack([-x_axis, -y_axis, approach_dir])
            try:
                q_alt_scipy = R.from_matrix(rot_mat_alt).as_quat()
                q_alt = (q_alt_scipy[3], q_alt_scipy[0],
                         q_alt_scipy[1], q_alt_scipy[2])
                dot_alt = abs(
                    q_current[0] * q_alt[0] + q_current[1] * q_alt[1]
                    + q_current[2] * q_alt[2] + q_current[3] * q_alt[3]
                )
                if dot_alt > dot:
                    q_target = q_alt
            except Exception:
                pass

        q_blend = quaternion_slerp(q_current, q_target, slerp_fraction)
        return Quaternion(w=q_blend[0], x=q_blend[1], y=q_blend[2], z=q_blend[3])

    # ------------------------------------------------------------------
    #  Contact-state classifier (force-based phase detection)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_contact_state(
        lateral_force: float,
        axial_force: float,
        baseline_lateral: float,
        baseline_axial: float,
        z_offset: float,
        prev_axial_forces: list,
        min_seated_depth: float = -0.010,
    ) -> int:
        """Classify the current contact state from force readings.

        Phase transitions form a principled state machine driven by F/T:
          FREE_SPACE    — no significant force change from baseline
          NEAR_CONTACT  — approaching surface (z_offset small, no contact yet)
          ALIGNMENT     — lateral force spike = plug touching board, not in port
          INSERTION     — axial force rising + lateral dropping = entering port
          SEATED        — axial force stabilised after min_seated_depth reached

        min_seated_depth is connector-aware: SFP needs -0.020, SC needs -0.035.
        Returns one of PHASE_FREE_SPACE .. PHASE_SEATED.
        """
        delta_lat = abs(lateral_force - baseline_lateral)
        delta_ax = abs(axial_force - baseline_axial)

        if z_offset < min_seated_depth:
            if delta_ax < FUNNEL_SEATED_Z_THRESHOLD and delta_lat < 2.0:
                return PHASE_SEATED
            return PHASE_INSERTION

        if z_offset < -0.005:
            return PHASE_INSERTION

        if delta_lat > FUNNEL_CONTACT_THRESHOLD:
            if delta_ax > FUNNEL_INSERTION_THRESHOLD and delta_lat < delta_ax * 0.5:
                return PHASE_INSERTION
            return PHASE_ALIGNMENT

        if z_offset < 0.02:
            return PHASE_NEAR_CONTACT

        return PHASE_FREE_SPACE

    def _get_phase_impedance(
        self, phase: int, residual_impedance: dict | None = None,
    ) -> dict:
        """Return impedance parameters for the given phase.

        If the residual MLP provides learned impedance modulation, blend it
        with the phase-specific profile.  The residual adds delta corrections
        on top of the phase base, giving the RL policy fine-grained control
        while the phase structure provides a safe operating envelope.
        """
        profile = PHASE_IMPEDANCE.get(phase, PHASE_IMPEDANCE[PHASE_FREE_SPACE])
        K = profile["K"].copy()
        D = profile["D"].copy()
        F_ff = np.zeros(6)

        if residual_impedance is not None:
            try:
                from training.residual_mlp import K_BASE, D_BASE
            except ImportError:
                K_BASE = np.array([90.0, 90.0, 40.0, 40.0, 40.0, 40.0])
                D_BASE = np.array([50.0, 50.0, 30.0, 20.0, 20.0, 20.0])
            delta_K = residual_impedance["stiffness"] - K_BASE
            delta_D = residual_impedance["damping"] - D_BASE
            K = np.clip(K + delta_K, 5.0, 300.0)
            D = np.clip(D + delta_D, 2.0, 150.0)
            F_ff = residual_impedance.get("feedforward_wrench", np.zeros(6))

        return {"stiffness": K, "damping": D, "feedforward_wrench": F_ff}

    def _force_funnel_correction(
        self,
        obs: Observation,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        baseline_lateral: float,
    ) -> np.ndarray:
        """Compute lateral correction using the force-funnel principle.

        When the plug contacts the board surface near (but not in) the port,
        lateral forces arise because the plug is pressing against the board
        at an offset from the opening. The direction opposite to the lateral
        force vector points toward the port opening — this is the "funnel"
        gradient.

        Unlike spiral search, this:
          - Only moves when contact is detected (safe)
          - Moves in a physically-grounded direction (toward opening)
          - Uses very low force / high compliance (won't damage components)
          - Converges in O(1) steps instead of O(n) spiral revolutions

        Returns corrected port_xyz (the target is shifted, not the robot).
        """
        w = obs.wrist_wrench.wrench
        F = np.array([w.force.x, w.force.y, w.force.z])

        # Project force onto the plane perpendicular to insertion axis
        F_along = np.dot(F, port_z_axis) * port_z_axis
        F_lateral = F - F_along
        lat_mag = np.linalg.norm(F_lateral)

        if lat_mag < FUNNEL_CONTACT_THRESHOLD:
            return port_xyz

        # Move target OPPOSITE to lateral force (toward the port opening)
        correction = -FUNNEL_GAIN * F_lateral
        correction_mag = np.linalg.norm(correction)
        if correction_mag > FUNNEL_MAX_CORRECTION:
            correction = correction * (FUNNEL_MAX_CORRECTION / correction_mag)

        return port_xyz + correction

    def _run_with_perception(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        port_transform: Transform,
        port_z_axis: np.ndarray,
    ) -> bool:
        """Continuous visual-servoing insertion with force-funnel refinement.

        Three control regimes, blended by proximity:
          1. Vision-dominant (z_offset > 5cm): V2 CNN re-detects every ~1s,
             EMA-smoothed updates steer the approach.
          2. Vision+Force (5cm > z_offset > 2cm): detection rate drops,
             force-funnel corrections added. EMA alpha decays with proximity.
          3. Force-dominant (z_offset < 2cm): vision updates stop,
             force-funnel and phase-adaptive impedance drive final insertion.
        """
        port_xyz = np.array([
            port_transform.translation.x,
            port_transform.translation.y,
            port_transform.translation.z,
        ])

        obs = get_observation()
        if obs is None:
            self.sleep_for(0.3)
            obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation for perception mode")
            return True

        baseline_fz = abs(self._get_fz(obs))
        baseline_lateral, baseline_z_abs, _ = self._get_force_magnitude(obs)
        self.get_logger().info(
            f"Perception mode: port=[{port_xyz[0]:.4f},{port_xyz[1]:.4f},{port_xyz[2]:.4f}] "
            f"axis=[{port_z_axis[0]:.3f},{port_z_axis[1]:.3f},{port_z_axis[2]:.3f}] "
            f"baseline_fz={baseline_fz:.1f}N baseline_lat={baseline_lateral:.1f}N"
        )

        # Graduated perception trust: stricter when far (3/3 cameras), relaxed
        # when close (accept 2 or even 1 camera with single-cam depth fallback).
        # The key insight: at close range, fewer cameras see the port, but the
        # one(s) that do have it filling their FOV → high resolution detection.
        initial_port_xyz = port_xyz.copy()
        vision_update_count = 0
        VISION_INTERVAL_APPROACH = 20
        VISION_INTERVAL_DESCENT = 15
        VISION_CUTOFF_Z = 0.04
        MAX_DRIFT_FROM_INITIAL = 0.025

        # =====================================================================
        # Phase 1: Approach with quality-gated visual servoing
        # =====================================================================
        send_feedback("Phase 1: approaching port (quality-gated visual servoing)")
        z_offset = 0.20
        n_approach = 100
        for t in range(n_approach):
            if self._time_remaining() <= 25:
                break
            frac = (t + 1) / n_approach
            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue

            # Graduated min_cams: strict at distance, relaxed closer in
            if t > 0 and t % VISION_INTERVAL_APPROACH == 0:
                if frac < 0.4:
                    req_cams, alpha_vis, max_shift_vis = 3, 0.50, 0.010
                elif frac < 0.8:
                    req_cams, alpha_vis, max_shift_vis = 2, 0.35, 0.008
                else:
                    req_cams, alpha_vis, max_shift_vis = 1, 0.20, 0.006

                det = self._detect_single_obs(obs, task, min_cams=req_cams)
                if det is not None:
                    new_xyz, new_axis, n_cams = det
                    drift = np.linalg.norm(new_xyz - initial_port_xyz)
                    if drift < MAX_DRIFT_FROM_INITIAL:
                        old_xyz = port_xyz.copy()
                        port_xyz = self._update_port_estimate(
                            port_xyz, new_xyz, alpha=alpha_vis, max_shift=max_shift_vis,
                        )
                        shift_mm = np.linalg.norm(port_xyz - old_xyz) * 1000
                        vision_update_count += 1
                        self.get_logger().info(
                            f"Approach vision #{vision_update_count} ({n_cams}/{req_cams}+ cams): "
                            f"shift={shift_mm:.1f}mm drift={drift*1000:.1f}mm α={alpha_vis} "
                            f"port=[{port_xyz[0]:.4f},{port_xyz[1]:.4f},{port_xyz[2]:.4f}]"
                        )
                    else:
                        self.get_logger().info(
                            f"Approach vision REJECTED: drift={drift*1000:.1f}mm > "
                            f"{MAX_DRIFT_FROM_INITIAL*1000:.0f}mm limit"
                        )

            base_target = self._calc_gripper_pose_from_observation(
                obs, port_xyz, port_z_axis, z_offset,
                position_fraction=frac,
                align_orientation=True,
                slerp_fraction=frac,
            )
            tcp = obs.controller_state.tcp_pose
            base_action_world = np.array([
                base_target.position.x - tcp.position.x,
                base_target.position.y - tcp.position.y,
                base_target.position.z - tcp.position.z,
                0.0, 0.0, 0.0,
            ])
            target, _imp = self._apply_residual(
                base_target, obs, port_xyz, port_z_axis, z_offset,
                base_action_world=base_action_world,
            )
            self._record_dagger_step(obs, base_target, port_xyz, port_z_axis, z_offset, base_action_world)
            self.set_pose_target(move_robot=move_robot, pose=target)
            self.sleep_for(0.05)

        # Settle at approach pose (no re-detection — cameras unreliable this close)
        for s in range(20):
            obs = get_observation()
            if obs is not None:
                base_target = self._calc_gripper_pose_from_observation(
                    obs, port_xyz, port_z_axis, z_offset,
                    align_orientation=True,
                )
                tcp = obs.controller_state.tcp_pose
                base_action_world = np.array([
                    base_target.position.x - tcp.position.x,
                    base_target.position.y - tcp.position.y,
                    base_target.position.z - tcp.position.z,
                    0.0, 0.0, 0.0,
                ])
                target, _imp = self._apply_residual(
                    base_target, obs, port_xyz, port_z_axis, z_offset,
                    base_action_world=base_action_world,
                )
                self._record_dagger_step(obs, base_target, port_xyz, port_z_axis, z_offset, base_action_world)
                self.set_pose_target(move_robot=move_robot, pose=target)
            self.sleep_for(0.05)

        # Update baseline forces
        obs = get_observation()
        if obs is not None:
            baseline_lateral, baseline_z_abs, _ = self._get_force_magnitude(obs)
            self.get_logger().info(
                f"Post-approach baseline: lat={baseline_lateral:.1f}N z={baseline_z_abs:.1f}N"
            )

        APPROACH_FORCE_ABORT = 15.0
        if baseline_lateral > APPROACH_FORCE_ABORT:
            self.get_logger().error(
                f"Post-approach lateral force {baseline_lateral:.1f}N too high, aborting"
            )
            self._collector.finish_episode(success=False)
            return True

        settled_offset = self._get_plug_offset(settled=True)
        self.get_logger().info(
            f"Approach complete, beginning descent. vision_updates={vision_update_count} "
            f"port=[{port_xyz[0]:.4f},{port_xyz[1]:.4f},{port_xyz[2]:.4f}] "
            f"plug_offset=[{settled_offset[0]:.4f},{settled_offset[1]:.4f},{settled_offset[2]:.4f}]"
        )

        is_sc_task = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()
        descent_limit = -0.045 if is_sc_task else -0.025
        min_seated_depth = -0.035 if is_sc_task else -0.020
        seated = False
        _seated_streak = 0

        for attempt in range(1, MAX_INSERTION_ATTEMPTS + 1):
            if self._time_remaining() <= RETRY_MIN_TIME and attempt > 1:
                self.get_logger().info(f"Not enough time for attempt {attempt}, skipping")
                break

            # =====================================================================
            # Phase 2: Descent with continuous perception + force-funnel
            # =====================================================================
            send_feedback(f"Phase 2: visual-servoing descent (attempt {attempt}/{MAX_INSERTION_ATTEMPTS})")
            self.get_logger().info(
                f"=== Insertion attempt {attempt}/{MAX_INSERTION_ATTEMPTS} "
                f"z_offset={z_offset:.4f} time_remaining={self._time_remaining():.0f}s ==="
            )
            self._current_attempt = attempt

            # Re-query perception on retry for a fresh port estimate
            if attempt > 1:
                det = self._detect_with_multi_obs(task, get_observation, n_obs=3, delay=0.15)
                if det is not None:
                    new_transform, new_axis = det
                    new_xyz = np.array([
                        new_transform.translation.x,
                        new_transform.translation.y,
                        new_transform.translation.z,
                    ])
                    drift = np.linalg.norm(new_xyz - initial_port_xyz)
                    if drift < 0.03:
                        old_xyz = port_xyz.copy()
                        port_xyz = 0.6 * new_xyz + 0.4 * port_xyz
                        shift_mm = np.linalg.norm(port_xyz - old_xyz) * 1000
                        self.get_logger().info(
                            f"Retry vision update: shift={shift_mm:.1f}mm "
                            f"port=[{port_xyz[0]:.4f},{port_xyz[1]:.4f},{port_xyz[2]:.4f}]"
                        )
                    else:
                        self.get_logger().info(
                            f"Retry vision rejected: drift={drift*1000:.1f}mm too large"
                        )

            # PPO retry penalty
            if attempt > 1 and self._ppo_train and self._ppo_trajectory:
                retry_penalty = -2.0
                self._ppo_trajectory[-1]["reward"] += retry_penalty
                self.get_logger().info(f"PPO: retry penalty {retry_penalty} applied")

            # Update force baselines
            _baseline_obs = get_observation()
            if _baseline_obs is not None:
                baseline_lateral, baseline_z_abs, _ = self._get_force_magnitude(_baseline_obs)

            force_pause_count = 0
            step = 0
            current_phase = PHASE_FREE_SPACE
            prev_phase = PHASE_FREE_SPACE
            axial_force_history: list[float] = []
            funnel_corrections = 0
            insertion_detected = False
            alignment_steps = 0
            spiral_attempted = False
            _seated_streak = 0
            if self._phase_hmm is not None:
                self._phase_hmm.reset()

            while z_offset > descent_limit:
                if self._time_remaining() <= 15:
                    self.get_logger().warn("Time low — stopping descent")
                    break

                if current_phase == PHASE_FREE_SPACE:
                    z_offset -= 0.0005
                elif current_phase == PHASE_NEAR_CONTACT:
                    z_offset -= 0.0003
                elif current_phase == PHASE_ALIGNMENT:
                    z_offset -= 0.0001
                elif current_phase == PHASE_INSERTION:
                    z_offset -= 0.0004
                elif current_phase == PHASE_SEATED:
                    break

                obs = get_observation()
                if obs is None:
                    self.sleep_for(0.05)
                    continue

                if (step % VISION_INTERVAL_DESCENT == 0
                        and z_offset > VISION_CUTOFF_Z
                        and current_phase < PHASE_INSERTION):
                    if z_offset > 0.12:
                        req_cams, max_alpha = 2, 0.40
                    elif z_offset > 0.06:
                        req_cams, max_alpha = 1, 0.25
                    else:
                        req_cams, max_alpha = 1, 0.15

                    det = self._detect_single_obs(obs, task, min_cams=req_cams)
                    if det is not None:
                        new_xyz, _, n_cams = det
                        drift = np.linalg.norm(new_xyz - initial_port_xyz)
                        if drift < MAX_DRIFT_FROM_INITIAL:
                            alpha = np.clip(
                                0.1 + (max_alpha - 0.1) * (z_offset - VISION_CUTOFF_Z) / 0.16,
                                0.1, max_alpha,
                            )
                            old_xyz = port_xyz.copy()
                            port_xyz = self._update_port_estimate(
                                port_xyz, new_xyz, alpha, max_shift=0.006,
                            )
                            shift_mm = np.linalg.norm(port_xyz - old_xyz) * 1000
                            vision_update_count += 1
                            if shift_mm > 0.3:
                                self.get_logger().info(
                                    f"Descent vision #{vision_update_count} ({n_cams}/{req_cams}+): "
                                    f"shift={shift_mm:.1f}mm α={alpha:.2f} z_off={z_offset:.4f}"
                                )

                lateral_f, z_f, total_f = self._get_force_magnitude(obs)
                axial_force_history.append(z_f)
                if len(axial_force_history) > 20:
                    axial_force_history.pop(0)

                if self._phase_hmm is not None:
                    _f_var = float(np.var(axial_force_history[-10:])) if len(axial_force_history) >= 2 else 0.0
                    _f_deriv = axial_force_history[-1] - axial_force_history[-2] if len(axial_force_history) >= 2 else 0.0
                    _hmm_obs = np.array([
                        abs(lateral_f - baseline_lateral), abs(z_f - baseline_z_abs),
                        z_offset, _f_deriv, _f_var,
                    ])
                    current_phase = self._phase_hmm.update(_hmm_obs, min_seated_depth=min_seated_depth)
                    _seated_streak = self._phase_hmm.seated_streak
                else:
                    _raw_phase = self._classify_contact_state(
                        lateral_f, z_f, baseline_lateral, baseline_z_abs,
                        z_offset, axial_force_history,
                        min_seated_depth=min_seated_depth,
                    )
                    if _raw_phase == PHASE_SEATED:
                        _seated_streak += 1
                        current_phase = PHASE_SEATED if _seated_streak >= 5 else PHASE_INSERTION
                    else:
                        _seated_streak = 0
                        current_phase = _raw_phase

                if current_phase != prev_phase:
                    phase_names = {
                        PHASE_FREE_SPACE: "FREE_SPACE",
                        PHASE_NEAR_CONTACT: "NEAR_CONTACT",
                        PHASE_ALIGNMENT: "ALIGNMENT",
                        PHASE_INSERTION: "INSERTION",
                        PHASE_SEATED: "SEATED",
                    }
                    _belief_str = self._phase_hmm.format_belief() if self._phase_hmm else ""
                    self.get_logger().info(
                        f"Phase: {phase_names[prev_phase]} → {phase_names[current_phase]} "
                        f"(z_off={z_offset:.4f} lat={lateral_f:.1f}N ax={z_f:.1f}N streak={_seated_streak}) {_belief_str}"
                    )
                    if current_phase == PHASE_INSERTION and not insertion_detected:
                        insertion_detected = True
                        self.get_logger().info(">>> Insertion detected — plug entering port")
                    prev_phase = current_phase

                if current_phase == PHASE_ALIGNMENT:
                    alignment_steps += 1
                else:
                    alignment_steps = 0

                if (alignment_steps >= 60
                        and not spiral_attempted
                        and not insertion_detected
                        and self._time_remaining() > 25):
                    spiral_attempted = True
                    send_feedback(f"Phase 2b: spiral search (attempt {attempt})")
                    self.get_logger().info(
                        f"Triggering spiral search (stuck ALIGNMENT for {alignment_steps} steps)"
                    )
                    z_offset, spiral_found = self._spiral_search(
                        get_observation, move_robot,
                        port_xyz, port_z_axis, z_offset,
                        baseline_lateral, baseline_z_abs,
                    )
                    if spiral_found:
                        insertion_detected = True
                        current_phase = PHASE_INSERTION
                        self.get_logger().info("Spiral found port — transitioning to INSERTION")
                    alignment_steps = 0

                delta_z = abs(z_f - baseline_z_abs)
                delta_lat = abs(lateral_f - baseline_lateral)

                if delta_z > FORCE_Z_LIMIT or delta_lat > FORCE_LATERAL_LIMIT:
                    force_pause_count += 1
                    retract = 0.002 if current_phase >= PHASE_ALIGNMENT else 0.003
                    z_offset += retract
                    self.get_logger().warn(
                        f"Force limit: Δlat={delta_lat:.1f} Δz={delta_z:.1f} "
                        f"— retract {retract*1000:.0f}mm (pause {force_pause_count}/{MAX_FORCE_PAUSES})"
                    )
                    self.sleep_for(0.1)
                    if force_pause_count >= MAX_FORCE_PAUSES:
                        self.get_logger().warn("Max force pauses — stopping descent")
                        break
                    continue
                force_pause_count = max(0, force_pause_count - 1)

                if current_phase >= PHASE_ALIGNMENT and current_phase < PHASE_SEATED:
                    port_xyz = self._force_funnel_correction(
                        obs, port_xyz, port_z_axis, baseline_lateral,
                    )
                    funnel_corrections += 1

                base_target = self._calc_gripper_pose_from_observation(
                    obs, port_xyz, port_z_axis, z_offset,
                    align_orientation=True,
                )
                tcp = obs.controller_state.tcp_pose
                base_action_world = np.array([
                    base_target.position.x - tcp.position.x,
                    base_target.position.y - tcp.position.y,
                    base_target.position.z - tcp.position.z,
                    0.0, 0.0, 0.0,
                ])
                force_deriv_ax = 0.0
                if len(axial_force_history) >= 2:
                    force_deriv_ax = axial_force_history[-1] - axial_force_history[-2]

                target, residual_impedance = self._apply_residual(
                    base_target, obs, port_xyz, port_z_axis, z_offset,
                    base_action_world=base_action_world,
                    phase=current_phase,
                    force_deriv_axial=force_deriv_ax,
                )
                self._record_dagger_step(obs, base_target, port_xyz, port_z_axis, z_offset, base_action_world)

                phase_imp = self._get_phase_impedance(current_phase, residual_impedance)
                self._set_pose_target_soft(
                    move_robot=move_robot, pose=target, impedance=phase_imp,
                )

                self.sleep_for(0.05)
                step += 1

                if step % 40 == 0:
                    _belief_log = f" belief={self._phase_hmm.format_belief()}" if self._phase_hmm else ""
                    self.get_logger().info(
                        f"Step {step}: z_off={z_offset:.4f} phase={current_phase} streak={_seated_streak} "
                        f"lat={lateral_f:.1f}N ax={z_f:.1f}N "
                        f"vis_updates={vision_update_count} funnel={funnel_corrections}{_belief_log}"
                    )

            # Check if insertion succeeded or is progressing well
            if current_phase >= PHASE_SEATED:
                seated = True
                self.get_logger().info(
                    f">>> SEATED on attempt {attempt}! z_offset={z_offset:.4f}"
                )
                break

            if current_phase >= PHASE_INSERTION:
                self.get_logger().info(
                    f"Attempt {attempt}: INSERTION phase reached (z_offset={z_offset:.4f}) "
                    f"— holding position (no retract needed)"
                )
                break

            # Clearly failed (FREE_SPACE / NEAR_CONTACT / ALIGNMENT) — retry
            if attempt < MAX_INSERTION_ATTEMPTS and self._time_remaining() > RETRY_MIN_TIME:
                self.get_logger().info(
                    f"Attempt {attempt} did NOT reach insertion (phase={current_phase}). "
                    f"Retracting for retry..."
                )
                send_feedback(f"Retracting for attempt {attempt + 1}")

                retract_target_z = RETRACT_HEIGHT
                retract_steps = int((retract_target_z - z_offset) / 0.001)
                for _ in range(min(retract_steps, 200)):
                    z_offset += 0.001
                    obs = get_observation()
                    if obs is not None:
                        target = self._calc_gripper_pose_from_observation(
                            obs, port_xyz, port_z_axis, z_offset,
                            align_orientation=True,
                        )
                        phase_imp = self._get_phase_impedance(PHASE_FREE_SPACE)
                        self._set_pose_target_soft(
                            move_robot=move_robot, pose=target, impedance=phase_imp,
                        )
                    self.sleep_for(0.03)

                for _ in range(RETRY_SETTLE_STEPS):
                    obs = get_observation()
                    if obs is not None:
                        target = self._calc_gripper_pose_from_observation(
                            obs, port_xyz, port_z_axis, z_offset,
                            align_orientation=True,
                        )
                        phase_imp = self._get_phase_impedance(PHASE_FREE_SPACE)
                        self._set_pose_target_soft(
                            move_robot=move_robot, pose=target, impedance=phase_imp,
                        )
                    self.sleep_for(0.05)

                self.get_logger().info(
                    f"Retracted to z_offset={z_offset:.4f}, ready for attempt {attempt + 1}"
                )
                self._tip_error_integrator = np.array([0.0, 0.0, 0.0])
            else:
                self.get_logger().info(
                    f"Attempt {attempt} failed (phase={current_phase}) — no more retries "
                    f"(time={self._time_remaining():.0f}s)"
                )

        # =====================================================================
        # Phase 3: Hold for scoring verification
        # =====================================================================
        send_feedback("Phase 3: holding (seated)" if seated else "Phase 3: holding (best effort)")
        self.get_logger().info(
            f"Hold at z_offset={z_offset:.4f} seated={seated} attempts_used={attempt}, "
            f"vision_updates={vision_update_count}, funnel_corrections={funnel_corrections}"
        )
        hold_secs = min(5.0, max(0.0, self._time_remaining() - 2.0))
        hold_steps = int(hold_secs / 0.05)
        for _ in range(hold_steps):
            obs = get_observation()
            if obs is not None:
                target = self._calc_gripper_pose_from_observation(
                    obs, port_xyz, port_z_axis, z_offset,
                    align_orientation=True,
                )
                phase_imp = self._get_phase_impedance(PHASE_SEATED)
                self._set_pose_target_soft(
                    move_robot=move_robot, pose=target, impedance=phase_imp,
                )
            self.sleep_for(0.05)

        self.get_logger().info(
            f"SmartInsert complete (Visual Servoing V5 + retry). "
            f"seated={seated}, attempts={attempt}, vision_updates={vision_update_count}"
        )
        return True

    def _spiral_search(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        port_xyz: np.ndarray,
        port_z_axis: np.ndarray,
        z_offset: float,
        baseline_lateral: float,
        baseline_z_abs: float,
    ) -> tuple[float, bool]:
        """Safe spiral search to find the port opening after surface contact.

        Sweeps in an expanding spiral perpendicular to the insertion axis while
        maintaining gentle downward pressure. Detects the port opening when
        axial force drops (plug slips in) and lateral force stays low.

        Safety guarantees:
          - Max radius: 3mm (won't wander far from estimated port)
          - Downward force capped: retracts if Δaxial > 5N
          - Lateral force limit: retracts if Δlateral > 6N
          - Very slow descent: 0.2mm per step only when forces are low
          - Total time: max 15 seconds (300 steps × 50ms)

        Returns (final_z_offset, insertion_found).
        """
        SPIRAL_MAX_RADIUS = 0.003    # 3mm — safe for SFP/SC ports
        SPIRAL_STEPS_PER_REV = 40    # 2 seconds per full revolution
        SPIRAL_RAMP_STEPS = 30       # ramp to max radius over 30 steps
        SPIRAL_MAX_STEPS = 300       # 15 seconds max
        SPIRAL_DESCENT_RATE = 0.0002 # 0.2mm per step when safe
        SPIRAL_SAFE_AX_DELTA = 5.0   # retract if axial force rise > 5N
        SPIRAL_SAFE_LAT_DELTA = 6.0  # retract if lateral force rise > 6N

        perp1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(perp1, port_z_axis)) > 0.9:
            perp1 = np.array([0.0, 1.0, 0.0])
        perp1 = perp1 - np.dot(perp1, port_z_axis) * port_z_axis
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(port_z_axis, perp1)

        self.get_logger().info(
            f"Spiral search: start z_off={z_offset:.4f} radius_max={SPIRAL_MAX_RADIUS*1000:.1f}mm"
        )

        insertion_found = False
        force_pauses = 0
        prev_ax = baseline_z_abs

        for i in range(SPIRAL_MAX_STEPS):
            if self._time_remaining() <= 12:
                break

            angle = 2.0 * np.pi * i / SPIRAL_STEPS_PER_REV
            radius = SPIRAL_MAX_RADIUS * min(1.0, i / SPIRAL_RAMP_STEPS)
            offset_xy = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            spiral_port = port_xyz + offset_xy

            obs = get_observation()
            if obs is None:
                self.sleep_for(0.05)
                continue

            lateral_f, z_f, _ = self._get_force_magnitude(obs)
            delta_ax = abs(z_f - baseline_z_abs)
            delta_lat = abs(lateral_f - baseline_lateral)

            # Safety: retract on excessive force
            if delta_ax > SPIRAL_SAFE_AX_DELTA or delta_lat > SPIRAL_SAFE_LAT_DELTA:
                force_pauses += 1
                z_offset += 0.001
                if force_pauses > 10:
                    self.get_logger().warn("Spiral: too many force pauses, stopping")
                    break
                self.sleep_for(0.1)
                continue
            force_pauses = max(0, force_pauses - 1)

            # Detect port opening: axial force drops while we're pressing,
            # AND lateral force is low (plug slipping into the hole)
            ax_drop = prev_ax - z_f
            if ax_drop > 2.0 and delta_lat < 2.0 and z_offset < 0.005:
                insertion_found = True
                self.get_logger().info(
                    f"Spiral: port opening detected at step {i}! "
                    f"ax_drop={ax_drop:.1f}N z_off={z_offset:.4f}"
                )
                # Push in gently
                for _ in range(40):
                    if z_offset > -0.015:
                        z_offset -= 0.0004
                    obs2 = get_observation()
                    if obs2 is not None:
                        tgt = self._calc_gripper_pose_from_observation(
                            obs2, spiral_port, port_z_axis, z_offset,
                            align_orientation=True,
                        )
                        phase_imp = self._get_phase_impedance(PHASE_INSERTION)
                        self._set_pose_target_soft(
                            move_robot=move_robot, pose=tgt, impedance=phase_imp,
                        )
                    self.sleep_for(0.05)
                break

            prev_ax = z_f

            # Gentle descent when forces are low
            if delta_ax < 2.0 and delta_lat < 2.0 and z_offset > -0.015:
                z_offset -= SPIRAL_DESCENT_RATE

            target = self._calc_gripper_pose_from_observation(
                obs, spiral_port, port_z_axis, z_offset,
                align_orientation=True,
            )
            phase_imp = self._get_phase_impedance(PHASE_ALIGNMENT)
            self._set_pose_target_soft(
                move_robot=move_robot, pose=target, impedance=phase_imp,
            )
            self.sleep_for(0.05)

            if i % 40 == 0:
                self.get_logger().info(
                    f"Spiral step {i}: r={radius*1000:.1f}mm z_off={z_offset:.4f} "
                    f"Δlat={delta_lat:.1f}N Δax={delta_ax:.1f}N"
                )

        self.get_logger().info(
            f"Spiral search complete: insertion={'YES' if insertion_found else 'NO'} "
            f"z_off={z_offset:.4f}"
        )
        return z_offset, insertion_found

    # ------------------------------------------------------------------
    #  Gripper-pose calculator  (heart of the TF-mode geometry)
    #
    #  Given the port's static transform and a desired offset along its
    #  insertion axis, compute where the gripper TCP should go so that the
    #  plug tip ends up at the right place.
    # ------------------------------------------------------------------

    def _calc_gripper_pose(
        self,
        port_transform: Transform,
        port_z_axis: Optional[np.ndarray] = None,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_integrator: bool = False,
    ) -> Pose:
        """Compute the gripper pose that aligns the plug tip with the port.

        This function is called every 50 ms during approach, descent and hold.
        It reads live TF each call so it adapts to the cable's flex in real time.

        Geometry overview (what each step does):
          1. Orientation alignment — quaternion slerp rotates the gripper so
             that the plug faces the same way as the port.
          2. Insertion-axis offset — positions the plug tip at a distance
             ``z_offset`` from the port center, along the port's own axis.
             This works for ANY port orientation (vertical SFP or horizontal SC).
          3. Cable-drift integrator — accumulates the sideways error between
             plug tip and port, and nudges the target to compensate.
          4. Position blending — during approach, ``position_fraction`` ramps
             from 0→1 so the gripper smoothly interpolates from its current
             position to the computed target.
        """
        if port_z_axis is None:
            port_z_axis = self._get_port_z_axis(port_transform)

        # ── Step 1: Orientation alignment ─────────────────────────────
        # Goal: rotate the gripper so the plug's orientation matches the port's.
        # Method: compute the rotation difference q_diff = q_port × q_plug⁻¹,
        #         then apply that correction to the gripper's current orientation,
        #         blending smoothly via slerp.
        q_port: QuaternionTuple = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )

        plug_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug: QuaternionTuple = (
            plug_tf.transform.rotation.w,
            plug_tf.transform.rotation.x,
            plug_tf.transform.rotation.y,
            plug_tf.transform.rotation.z,
        )
        # Inverse of plug quaternion: negate w (conjugate for unit quaternions).
        q_plug_inv: QuaternionTuple = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])

        # q_diff tells us "how much to rotate from plug orientation to port orientation".
        q_diff = quaternion_multiply(q_port, q_plug_inv)

        gripper_tf = self._parent_node._tf_buffer.lookup_transform(
            "base_link", "gripper/tcp", Time()
        )
        q_gripper: QuaternionTuple = (
            gripper_tf.transform.rotation.w,
            gripper_tf.transform.rotation.x,
            gripper_tf.transform.rotation.y,
            gripper_tf.transform.rotation.z,
        )
        # Apply the same rotation correction to the gripper.
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)

        # Smoothly blend from current orientation → target orientation.
        q_blend = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        # ── Step 2: Position along insertion axis ─────────────────────
        # Read current 3-D positions of gripper TCP and plug tip.
        g = gripper_tf.transform.translation
        gripper_xyz = np.array([g.x, g.y, g.z])

        p = plug_tf.transform.translation
        plug_xyz = np.array([p.x, p.y, p.z])

        port_xyz = np.array([
            port_transform.translation.x,
            port_transform.translation.y,
            port_transform.translation.z,
        ])

        # Where the plug tip should be:
        #   port center  −  z_offset × port_Z_axis
        #
        # port_Z_axis points INTO the port.  Subtracting means:
        #   z_offset > 0  →  tip is on the approach side (before the port)
        #   z_offset < 0  →  tip is past the center (plug is inserted)
        plug_tip_target = port_xyz - z_offset * port_z_axis

        # ── Step 3: Cable-drift integrator ────────────────────────────
        # Measure how far the plug tip has drifted from the port center,
        # but only the component PERPENDICULAR to the insertion axis
        # (drift along the insertion axis is handled by z_offset).
        total_error = port_xyz - plug_xyz
        along_axis = np.dot(total_error, port_z_axis) * port_z_axis
        lateral_error = total_error - along_axis

        if reset_integrator:
            self._tip_error_integrator = np.array([0.0, 0.0, 0.0])
        else:
            self._tip_error_integrator = np.clip(
                self._tip_error_integrator + lateral_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        # Nudge the plug-tip target sideways to compensate for drift.
        plug_tip_target = plug_tip_target + self._i_gain * self._tip_error_integrator

        # ── Step 4: Convert plug-tip target → gripper target ──────────
        # The gripper and plug tip are connected by the cable + gripper fingers.
        # We use CheatCode's feedback-control trick: subtract the current
        # gripper-to-plug offset.  Over multiple iterations this converges
        # because the offset is re-read from live TF each time.
        gripper_plug_offset = gripper_xyz - plug_xyz
        gripper_target = plug_tip_target - gripper_plug_offset

        # Wrist clearance for SC horizontal insertion (same as perception mode)
        is_sc = self._task and "sc" in str(getattr(self._task, 'port_type', '') or '').lower()
        if is_sc and abs(port_z_axis[2]) < 0.5:
            clearance_ramp = min(1.0, max(0.0, z_offset / 0.05))
            gripper_target[2] += SC_WRIST_CLEARANCE_Z * clearance_ramp

        # ── Step 5: Smooth blending during approach ───────────────────
        # position_fraction ramps 0→1 during the approach phase, causing
        # the gripper to glide from its current pose to the computed target.
        blend = gripper_xyz + position_fraction * (gripper_target - gripper_xyz)

        return Pose(
            position=Point(x=float(blend[0]), y=float(blend[1]), z=float(blend[2])),
            orientation=Quaternion(
                w=q_blend[0], x=q_blend[1], y=q_blend[2], z=q_blend[3]
            ),
        )

    # ==================================================================
    #  MODE 2:  Blind descent  (ground_truth:=false / evaluation)
    #
    #  No TF knowledge of the port.  We assume the gripper is already
    #  roughly positioned (e.g. by a camera-based approach — future work).
    #  Strategy: descend slowly from the current TCP, using the force-
    #  torque sensor to detect contact and make small lateral corrections.
    # ==================================================================

    def _run_blind_descent(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback("Blind descent — no ground truth")

        # Get current state from the observation (joint poses, wrench, etc.).
        obs = get_observation()
        if obs is None:
            self.sleep_for(0.5)
            obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation available")
            return True

        # Read the gripper's current pose (the "TCP" — Tool Center Point).
        tcp = obs.controller_state.tcp_pose
        current_x = tcp.position.x
        current_y = tcp.position.y
        current_z = tcp.position.z

        # Record the baseline force so we only react to CHANGES (the cable's
        # weight creates a constant ~19 N that we want to ignore).
        baseline_fx = obs.wrist_wrench.wrench.force.x
        baseline_fy = obs.wrist_wrench.wrench.force.y
        baseline_fz = obs.wrist_wrench.wrench.force.z
        self.get_logger().info(
            f"Blind descent from x={current_x:.4f} y={current_y:.4f} z={current_z:.4f}"
        )
        self.get_logger().info(
            f"Baseline force: fx={baseline_fx:.1f} fy={baseline_fy:.1f} fz={baseline_fz:.1f}"
        )

        force_pause_count = 0
        total_descent = 0.0
        max_descent = 0.05  # stop after 50 mm

        # --- Phase 1: Slow descent with force-guided X-Y correction -----------
        send_feedback("Phase 1: descending with force feedback")
        step = 0
        while total_descent < max_descent:
            if self._time_remaining() <= 15:
                self.get_logger().warn("Time low — stopping descent")
                break

            # Move down 0.5 mm.
            current_z -= 0.0005
            total_descent += 0.0005

            target = Pose(
                position=Point(x=current_x, y=current_y, z=current_z),
                orientation=tcp.orientation,
            )
            self.set_pose_target(move_robot=move_robot, pose=target)
            self.sleep_for(0.05)

            # Read fresh force data.
            obs = get_observation()
            if obs is None:
                continue

            fx = obs.wrist_wrench.wrench.force.x
            fy = obs.wrist_wrench.wrench.force.y
            fz = obs.wrist_wrench.wrench.force.z

            # Compute force CHANGE from baseline (ignoring cable weight).
            dfx = fx - baseline_fx
            dfy = fy - baseline_fy
            dfz = fz - baseline_fz

            # If the Z-force spiked, back off and pause.
            if abs(dfz) > FORCE_CHANGE_LIMIT:
                force_pause_count += 1
                self.get_logger().warn(
                    f"Force change dfz={dfz:.1f}N "
                    f"(pause {force_pause_count}/{MAX_FORCE_PAUSES})"
                )
                current_z += 0.001
                total_descent -= 0.001
                self.sleep_for(0.1)
                if force_pause_count > MAX_FORCE_PAUSES:
                    self.get_logger().warn("Max force pauses — stopping")
                    break
            else:
                force_pause_count = max(0, force_pause_count - 1)

            # Small lateral corrections opposite to lateral forces.
            correction_gain = 0.0001
            if abs(dfx) > 2.0:
                current_x -= correction_gain * np.sign(dfx)
            if abs(dfy) > 2.0:
                current_y -= correction_gain * np.sign(dfy)

            step += 1
            if step % 20 == 0:
                self.get_logger().info(
                    f"Descent {total_descent*1000:.1f}mm  "
                    f"dfx={dfx:.1f} dfy={dfy:.1f} dfz={dfz:.1f}"
                )

        # --- Phase 2: Hold for scoring verification --------------------------
        send_feedback("Phase 2: holding for verification")
        self.get_logger().info(
            f"Holding at z={current_z:.4f} (descended {total_descent*1000:.1f}mm)"
        )
        hold_secs = min(5.0, max(0.0, self._time_remaining() - 2.0))
        hold_steps = int(hold_secs / 0.05)
        for _ in range(hold_steps):
            target = Pose(
                position=Point(x=current_x, y=current_y, z=current_z),
                orientation=tcp.orientation,
            )
            self.set_pose_target(move_robot=move_robot, pose=target)
            self.sleep_for(0.05)

        self.get_logger().info("SmartInsert complete (blind descent)")
        return True

    # ==================================================================
    #  SHARED HELPER METHODS
    # ==================================================================

    @staticmethod
    def _get_port_z_axis(port_transform: Transform) -> np.ndarray:
        """Return the port's local Z-axis direction as a unit vector in base_link.

        The port's Z-axis points INTO the port (the direction the plug travels
        during insertion).  We derive it by rotating the unit vector [0,0,1]
        by the port's orientation quaternion.

        Why this matters:
          - SFP ports face upward  →  Z ≈ [0, 0, −1]  (downward in world)
          - SC  ports face sideways →  Z ≈ [horizontal]
        By computing this dynamically, the same code handles every port type.
        """
        q = port_transform.rotation
        w, x, y, z = q.w, q.x, q.y, q.z

        # Standard quaternion rotation of [0, 0, 1]:
        zx = 2.0 * (x * z + w * y)
        zy = 2.0 * (y * z - w * x)
        zz = 1.0 - 2.0 * (x * x + y * y)

        axis = np.array([zx, zy, zz])

        # Normalise for safety (should already be unit length).
        length = np.linalg.norm(axis)
        if length > 1e-6:
            axis /= length
        return axis

    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 5.0
    ) -> bool:
        """Block until a TF frame becomes available, or time out.

        Uses wall-clock as a safety net so that a frozen sim clock cannot
        cause an infinite hang.
        """
        import time as _time
        wall_start = _time.monotonic()
        wall_limit = max(timeout_sec * 3, 15.0)

        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while True:
            sim_elapsed = (self.time_now() - start) if start.nanoseconds > 0 else Duration(seconds=0)
            wall_elapsed = _time.monotonic() - wall_start
            if sim_elapsed >= timeout or wall_elapsed > wall_limit:
                break
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame, source_frame, Time()
                )
                print(f"[SmartInsert] TF found: {source_frame}", flush=True)
                self.get_logger().info(f"TF found: {source_frame}")
                return True
            except TransformException:
                if attempt % 10 == 0:
                    print(f"[SmartInsert] Waiting for TF: {source_frame} (wall={wall_elapsed:.1f}s)", flush=True)
                    self.get_logger().info(f"Waiting for TF: {source_frame}...")
                attempt += 1
                _time.sleep(0.5)
        print(f"[SmartInsert] TF not available: {source_frame} (wall={_time.monotonic()-wall_start:.1f}s)", flush=True)
        self.get_logger().warn(
            f"TF not available: {source_frame} (after {timeout_sec}s)"
        )
        return False

    @staticmethod
    def _get_fz(obs: Observation) -> float:
        """Safely extract the Z-axis force from an observation."""
        try:
            return obs.wrist_wrench.wrench.force.z
        except Exception:
            return 0.0

    def _time_remaining(self) -> float:
        """Seconds of sim-time left before our self-imposed budget expires."""
        elapsed_ns = self.time_now().nanoseconds - self._start_time_ns
        return max(0.0, self._time_budget - elapsed_ns / 1e9)
