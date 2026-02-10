"""
============================================================================
UR5e CABLE INSERTION ENVIRONMENT - AI for Industry Challenge (Preliminary)
============================================================================

Preliminary environment for the AI for Industry Challenge.
https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380

This is a practice environment created BEFORE the official toolkit release
(March 2, 2026). Once the toolkit is released, this will be updated with
the official assets and specifications.

CHALLENGE HARDWARE STACK:
- Arm: Universal Robots UR5e
- Gripper: Robotiq Hand-E
- Sensor: Axia80 Force-Torque sensor
- Vision: Three wrist-mounted Basler cameras

TASK: Cable insertion and deformable object manipulation

Author: Ashutosh Kumar
============================================================================
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# ROBOT CONFIGURATION (UR5e Specifications)
# =============================================================================

@dataclass
class UR5eConfig:
    """
    Universal Robots UR5e specifications.
    
    Official specs: https://www.universal-robots.com/products/ur5-robot/
    """
    # Joint limits (radians)
    joint_limits_lower: np.ndarray = None
    joint_limits_upper: np.ndarray = None
    
    # Velocity limits (rad/s)
    max_joint_velocity: float = 3.14  # 180 deg/s
    
    # Payload
    max_payload_kg: float = 5.0
    
    # Reach
    reach_mm: float = 850.0
    
    # Repeatability
    repeatability_mm: float = 0.03
    
    # Degrees of freedom
    dof: int = 6
    
    def __post_init__(self):
        # UR5e joint limits (all joints: ±360°)
        self.joint_limits_lower = np.array([-2*np.pi] * 6)
        self.joint_limits_upper = np.array([2*np.pi] * 6)


@dataclass
class RobotiqHandEConfig:
    """
    Robotiq Hand-E gripper specifications.
    
    Official specs: https://robotiq.com/products/hand-e-adaptive-robot-gripper
    """
    # Stroke (opening width)
    stroke_mm: float = 50.0
    
    # Grip force range
    min_grip_force_n: float = 20.0
    max_grip_force_n: float = 185.0
    
    # Speed
    max_speed_mm_s: float = 150.0
    
    # Payload (what it can lift)
    max_payload_kg: float = 5.0


@dataclass 
class Axia80FTConfig:
    """
    ATI Axia80 Force-Torque sensor specifications.
    """
    # Force sensing range (N)
    fx_fy_range_n: float = 800.0
    fz_range_n: float = 2400.0
    
    # Torque sensing range (Nm)
    tx_ty_range_nm: float = 40.0
    tz_range_nm: float = 40.0
    
    # Resolution
    force_resolution_n: float = 0.25
    torque_resolution_nm: float = 0.0125


# =============================================================================
# CABLE INSERTION TASK
# =============================================================================

@dataclass
class CableInsertionTask:
    """
    Defines a cable insertion task for the challenge.
    
    The challenge involves inserting various cable types into ports,
    handling deformable objects (cables/wires).
    """
    # Cable properties
    cable_type: str = "ethernet"  # ethernet, usb, power, etc.
    cable_length_mm: float = 300.0
    cable_diameter_mm: float = 6.0
    cable_stiffness: float = 0.5  # 0=very flexible, 1=rigid
    
    # Port/connector properties
    port_type: str = "rj45"  # rj45, usb-a, usb-c, etc.
    port_position: np.ndarray = None  # [x, y, z] in meters
    port_orientation: np.ndarray = None  # quaternion [w, x, y, z]
    
    # Insertion tolerances
    position_tolerance_mm: float = 2.0
    angle_tolerance_deg: float = 5.0
    
    # Success criteria
    required_insertion_depth_mm: float = 15.0
    max_insertion_force_n: float = 20.0
    
    def __post_init__(self):
        if self.port_position is None:
            self.port_position = np.array([0.4, 0.0, 0.2])  # Default position
        if self.port_orientation is None:
            self.port_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity


# =============================================================================
# ENVIRONMENT (Gymnasium-style interface)
# =============================================================================

class UR5eCableInsertionEnv:
    """
    Preliminary environment for UR5e cable insertion task.
    
    This provides a Gymnasium-style interface for:
    - State observation (joint positions, gripper state, F/T sensor, cameras)
    - Action commands (joint velocities or end-effector pose)
    - Reward computation
    
    NOTE: This is a PLACEHOLDER until the official toolkit is released.
    The actual simulation will use Isaac Sim with proper physics.
    """
    
    def __init__(
        self,
        task: Optional[CableInsertionTask] = None,
        control_mode: str = "joint_velocity",  # or "end_effector"
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.
        
        Args:
            task: Cable insertion task configuration
            control_mode: "joint_velocity" or "end_effector"
            render_mode: "human", "rgb_array", or None
        """
        self.task = task or CableInsertionTask()
        self.control_mode = control_mode
        self.render_mode = render_mode
        
        # Robot configurations
        self.ur5e = UR5eConfig()
        self.gripper = RobotiqHandEConfig()
        self.ft_sensor = Axia80FTConfig()
        
        # State
        self.joint_positions = np.zeros(6)
        self.joint_velocities = np.zeros(6)
        self.gripper_position = 0.0  # 0=closed, 1=open
        self.ft_reading = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]
        
        # Episode tracking
        self.step_count = 0
        self.max_steps = 1000
        
        # Observation and action spaces (will be properly defined with gymnasium)
        self.observation_dim = 6 + 6 + 1 + 6  # joints + velocities + gripper + F/T
        self.action_dim = 6 + 1  # joint velocities + gripper
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to home position
        self.joint_positions = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
        self.joint_velocities = np.zeros(6)
        self.gripper_position = 1.0  # Open
        self.ft_reading = np.zeros(6)
        self.step_count = 0
        
        obs = self._get_observation()
        info = {"task": self.task.cable_type, "port": self.task.port_type}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint velocities (6) + gripper command (1)
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: Episode ended (success or failure)
            truncated: Episode truncated (max steps)
            info: Additional information
        """
        # Parse action
        joint_vel_command = action[:6]
        gripper_command = action[6] if len(action) > 6 else 0.0
        
        # Apply action (simplified - actual physics in Isaac Sim)
        dt = 0.01  # 100 Hz control
        self.joint_velocities = np.clip(
            joint_vel_command, 
            -self.ur5e.max_joint_velocity, 
            self.ur5e.max_joint_velocity
        )
        self.joint_positions += self.joint_velocities * dt
        self.joint_positions = np.clip(
            self.joint_positions,
            self.ur5e.joint_limits_lower,
            self.ur5e.joint_limits_upper
        )
        
        # Update gripper
        self.gripper_position = np.clip(gripper_command, 0.0, 1.0)
        
        # Simulate F/T sensor (placeholder)
        self.ft_reading = np.random.randn(6) * 0.1  # Noise
        
        self.step_count += 1
        
        # Compute reward (placeholder - will be task-specific)
        reward = self._compute_reward()
        
        # Check termination
        terminated = self._check_success()
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "step": self.step_count,
            "success": terminated and reward > 0
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            [self.gripper_position],
            self.ft_reading
        ])
    
    def _compute_reward(self) -> float:
        """
        Compute reward for current state.
        
        TODO: Implement proper reward based on:
        - Distance to target port
        - Cable alignment
        - Insertion depth
        - Force feedback (avoid excessive force)
        """
        # Placeholder: small negative reward per step (encourage efficiency)
        return -0.01
    
    def _check_success(self) -> bool:
        """
        Check if insertion task is complete.
        
        TODO: Implement proper success check based on:
        - Cable fully inserted
        - Proper connection detected
        """
        return False  # Placeholder
    
    def render(self):
        """
        Render the environment.
        
        NOTE: Actual rendering will be done in Isaac Sim.
        """
        if self.render_mode == "human":
            print(f"Step {self.step_count}: joints={self.joint_positions[:3]}...")
    
    def close(self):
        """Clean up resources."""
        pass


# =============================================================================
# ISAAC SIM INTEGRATION
# =============================================================================

# Correct asset paths discovered from Isaac Sim Content Browser
ISAAC_SIM_ASSETS = {
    "ur5e": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "ur10e": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd",
}


def create_isaac_sim_env(headless: bool = False):
    """
    Create the UR5e cable insertion environment in Isaac Sim.
    
    Args:
        headless: Run without GUI (for training)
        
    Returns:
        world: Isaac Sim World object
        ur5e_prim_path: Path to UR5e robot in stage
    """
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})
    
    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.objects import FixedCuboid
    
    # Create world
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Load UR5e robot
    ur5e_path = ISAAC_SIM_ASSETS["ur5e"]
    print(f"Loading UR5e from: {ur5e_path}")
    add_reference_to_stage(usd_path=ur5e_path, prim_path="/World/UR5e")
    
    # Add table
    table = FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.5, 0.0, 0.4]),
        scale=np.array([0.6, 0.8, 0.02]),
        color=np.array([0.6, 0.4, 0.2])
    )
    world.scene.add(table)
    
    # Add server tray (insertion target)
    server = FixedCuboid(
        prim_path="/World/ServerTray",
        name="server_tray",
        position=np.array([0.5, 0.0, 0.43]),
        scale=np.array([0.4, 0.3, 0.04]),
        color=np.array([0.15, 0.15, 0.15])
    )
    world.scene.add(server)
    
    # Add ethernet port (yellow highlight)
    port = FixedCuboid(
        prim_path="/World/Port",
        name="ethernet_port",
        position=np.array([0.55, 0.1, 0.46]),
        scale=np.array([0.02, 0.015, 0.012]),
        color=np.array([1.0, 0.8, 0.0])
    )
    world.scene.add(port)
    
    # Add cable placeholder (blue)
    cable = FixedCuboid(
        prim_path="/World/Cable",
        name="cable",
        position=np.array([0.3, -0.15, 0.43]),
        scale=np.array([0.15, 0.008, 0.008]),
        color=np.array([0.0, 0.3, 0.8])
    )
    world.scene.add(cable)
    
    world.reset()
    
    return world, simulation_app, "/World/UR5e"


def save_scene_as_usd(output_path: str):
    """
    Save the current Isaac Sim scene as a USD file.
    
    Args:
        output_path: Path to save the USD file
    """
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(output_path)
    print(f"Scene saved to: {output_path}")


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("UR5e Cable Insertion Environment (Preliminary)")
    print("=" * 60)
    print()
    
    # Create environment
    task = CableInsertionTask(
        cable_type="ethernet",
        port_type="rj45",
        cable_length_mm=300.0
    )
    
    env = UR5eCableInsertionEnv(task=task, render_mode="human")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Task: {info}")
    print()
    
    # Test a few steps with random actions
    print("Running 5 random steps...")
    for i in range(5):
        action = np.random.randn(7) * 0.1  # Random small actions
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, done={terminated or truncated}")
    
    print()
    print("=" * 60)
    print("NOTE: This is a PRELIMINARY environment.")
    print("Official toolkit releases March 2, 2026.")
    print("=" * 60)
