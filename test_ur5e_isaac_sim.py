"""
============================================================================
TEST UR5e IN ISAAC SIM - Basic Verification
============================================================================

Test script to verify UR5e robot setup in Isaac Sim before pushing to git.

Run with Isaac Sim Python:
    C:\isaacsim\IsaacLab\_isaac_sim\python.bat test_ur5e_isaac_sim.py

Author: Ashutosh Kumar
============================================================================
"""

def main():
    # =========================================================================
    # STEP 1: Launch Isaac Sim
    # =========================================================================
    print("=" * 60)
    print("UR5e Isaac Sim Test")
    print("=" * 60)
    print()
    print("Launching Isaac Sim...")
    
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    print("Isaac Sim launched successfully!")
    print()
    
    # =========================================================================
    # STEP 2: Import Isaac Sim modules
    # =========================================================================
    import numpy as np
    from omni.isaac.core import World
    from omni.isaac.core.robots import Robot
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.objects import FixedCuboid, VisualCuboid
    
    # =========================================================================
    # STEP 3: Create World
    # =========================================================================
    print("Creating world...")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # =========================================================================
    # STEP 4: Load UR5e Robot
    # =========================================================================
    print("Loading UR5e robot...")
    
    # Get Isaac Sim assets path
    assets_root = get_assets_root_path()
    
    # UR5e USD path in Isaac Sim assets
    # Correct path discovered from Content Browser
    ur5e_paths = [
        # Isaac Sim 5.1 (current version)
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        # Alternative paths
        f"{assets_root}/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        f"{assets_root}/Isaac/Robots/UR5e/ur5e.usd",
    ]
    
    ur5e_loaded = False
    for ur5e_path in ur5e_paths:
        try:
            print(f"  Trying: {ur5e_path}")
            add_reference_to_stage(usd_path=ur5e_path, prim_path="/World/UR5e")
            ur5e_loaded = True
            print(f"  SUCCESS: Loaded UR5e from {ur5e_path}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
    
    if not ur5e_loaded:
        print("\nWARNING: Could not load UR5e from standard paths.")
        print("Creating a placeholder robot visualization instead...")
        
        # Create placeholder arm visualization
        base = FixedCuboid(
            prim_path="/World/UR5e_Placeholder/Base",
            name="ur5e_base",
            position=np.array([0.0, 0.0, 0.05]),
            scale=np.array([0.2, 0.2, 0.1]),
            color=np.array([0.3, 0.3, 0.3])
        )
        world.scene.add(base)
        
        # Simple arm representation
        link1 = FixedCuboid(
            prim_path="/World/UR5e_Placeholder/Link1",
            name="link1",
            position=np.array([0.0, 0.0, 0.25]),
            scale=np.array([0.08, 0.08, 0.3]),
            color=np.array([0.2, 0.4, 0.8])
        )
        world.scene.add(link1)
        
        link2 = FixedCuboid(
            prim_path="/World/UR5e_Placeholder/Link2",
            name="link2",
            position=np.array([0.0, 0.2, 0.4]),
            scale=np.array([0.06, 0.3, 0.06]),
            color=np.array([0.2, 0.4, 0.8])
        )
        world.scene.add(link2)
        
        # End effector placeholder
        gripper = FixedCuboid(
            prim_path="/World/UR5e_Placeholder/Gripper",
            name="gripper",
            position=np.array([0.0, 0.4, 0.4]),
            scale=np.array([0.1, 0.05, 0.15]),
            color=np.array([0.5, 0.5, 0.5])
        )
        world.scene.add(gripper)
    
    # =========================================================================
    # STEP 5: Add Task Scene Elements
    # =========================================================================
    print("\nAdding task scene elements...")
    
    # Table/workstation
    table = FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.5, 0.0, 0.35]),
        scale=np.array([0.6, 0.8, 0.02]),
        color=np.array([0.6, 0.4, 0.2])
    )
    world.scene.add(table)
    
    # Server tray (target for cable insertion)
    server_tray = FixedCuboid(
        prim_path="/World/ServerTray",
        name="server_tray",
        position=np.array([0.5, 0.0, 0.38]),
        scale=np.array([0.4, 0.3, 0.04]),
        color=np.array([0.2, 0.2, 0.2])
    )
    world.scene.add(server_tray)
    
    # Port/connector target (RJ45 port placeholder)
    port = FixedCuboid(
        prim_path="/World/Port",
        name="ethernet_port",
        position=np.array([0.5, 0.1, 0.41]),
        scale=np.array([0.02, 0.015, 0.015]),
        color=np.array([0.8, 0.8, 0.0])  # Yellow to highlight
    )
    world.scene.add(port)
    
    # Cable placeholder (will be deformable body in full sim)
    cable_start = FixedCuboid(
        prim_path="/World/Cable",
        name="cable",
        position=np.array([0.3, -0.1, 0.38]),
        scale=np.array([0.15, 0.01, 0.01]),
        color=np.array([0.0, 0.0, 0.8])  # Blue cable
    )
    world.scene.add(cable_start)
    
    print("  Added: Table, Server Tray, Port, Cable placeholder")
    
    # =========================================================================
    # STEP 6: Add Labels
    # =========================================================================
    print("\nScene setup complete!")
    print()
    print("=" * 60)
    print("SCENE ELEMENTS:")
    print("=" * 60)
    print("  - UR5e Robot (or placeholder)")
    print("  - Work Table")
    print("  - Server Tray (insertion target)")
    print("  - Ethernet Port (yellow)")
    print("  - Cable (blue)")
    print()
    print("=" * 60)
    print("CONTROLS:")
    print("=" * 60)
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - WASD: Move camera")
    print("  - Close window to exit")
    print("=" * 60)
    
    # =========================================================================
    # STEP 7: Run Simulation
    # =========================================================================
    world.reset()
    
    print("\nRunning simulation... (close window to exit)")
    
    step_count = 0
    while simulation_app.is_running():
        world.step(render=True)
        step_count += 1
        
        # Print status every 500 steps
        if step_count % 500 == 0:
            print(f"  Simulation step: {step_count}")
    
    # =========================================================================
    # STEP 8: Cleanup
    # =========================================================================
    print("\nClosing Isaac Sim...")
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
