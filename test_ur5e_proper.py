"""
============================================================================
TEST UR5e IN ISAAC SIM - Proper Robot Loading
============================================================================

Loads the actual UR5e robot from Isaac Sim's Nucleus server.

Run with Isaac Sim Python:
    C:\isaacsim\IsaacLab\_isaac_sim\python.bat test_ur5e_proper.py

Author: Ashutosh Kumar
============================================================================
"""

def main():
    print("=" * 60)
    print("UR5e Isaac Sim Test - Proper Robot")
    print("=" * 60)
    print()
    print("Launching Isaac Sim...")
    
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    print("Isaac Sim launched!")
    
    import numpy as np
    import omni.usd
    from pxr import UsdGeom, Gf, Sdf
    
    from omni.isaac.core import World
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.objects import FixedCuboid
    
    # Create world
    print("\nCreating world...")
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Get Nucleus asset path
    assets_root = get_assets_root_path()
    print(f"Assets root: {assets_root}")
    
    # =========================================================================
    # TRY MULTIPLE UR5e PATHS
    # =========================================================================
    
    ur5e_asset_paths = [
        # Isaac Sim 4.x paths
        f"{assets_root}/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        f"{assets_root}/Isaac/Robots/UR5e/ur5e.usd",
        f"{assets_root}/Isaac/Robots/Universal_Robots/ur5e/ur5e.usd",
        # Older paths
        "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    ]
    
    print("\n" + "=" * 60)
    print("LOADING UR5e ROBOT")
    print("=" * 60)
    
    ur5e_loaded = False
    for path in ur5e_asset_paths:
        try:
            print(f"\nTrying: {path}")
            add_reference_to_stage(usd_path=path, prim_path="/World/UR5e")
            ur5e_loaded = True
            print(f"SUCCESS! Loaded UR5e from:\n  {path}")
            break
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}...")
    
    if not ur5e_loaded:
        print("\n" + "!" * 60)
        print("Could not load UR5e from standard paths.")
        print("Let's try browsing what's available...")
        print("!" * 60)
        
        # List available robots
        try:
            import omni.client
            result, entries = omni.client.list(f"{assets_root}/Isaac/Robots/")
            print(f"\nAvailable in {assets_root}/Isaac/Robots/:")
            for entry in entries[:20]:  # First 20
                print(f"  - {entry.relative_path}")
        except Exception as e:
            print(f"Could not list: {e}")
        
        print("\nCreating placeholder robot instead...")
        # Placeholder
        base = FixedCuboid(
            prim_path="/World/UR5e_Placeholder",
            name="ur5e_placeholder",
            position=np.array([0.0, 0.0, 0.3]),
            scale=np.array([0.15, 0.15, 0.6]),
            color=np.array([0.2, 0.4, 0.8])
        )
        world.scene.add(base)
    
    # =========================================================================
    # ADD SCENE ELEMENTS
    # =========================================================================
    
    print("\nAdding scene elements...")
    
    # Table
    table = FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.5, 0.0, 0.4]),
        scale=np.array([0.6, 0.8, 0.02]),
        color=np.array([0.6, 0.4, 0.2])
    )
    world.scene.add(table)
    
    # Server tray
    server = FixedCuboid(
        prim_path="/World/ServerTray",
        name="server",
        position=np.array([0.5, 0.0, 0.43]),
        scale=np.array([0.4, 0.3, 0.04]),
        color=np.array([0.15, 0.15, 0.15])
    )
    world.scene.add(server)
    
    # Ethernet port (yellow highlight)
    port = FixedCuboid(
        prim_path="/World/Port",
        name="port",
        position=np.array([0.55, 0.1, 0.46]),
        scale=np.array([0.02, 0.015, 0.012]),
        color=np.array([1.0, 0.8, 0.0])
    )
    world.scene.add(port)
    
    # Cable (blue)
    cable = FixedCuboid(
        prim_path="/World/Cable",
        name="cable",
        position=np.array([0.3, -0.15, 0.43]),
        scale=np.array([0.15, 0.008, 0.008]),
        color=np.array([0.0, 0.3, 0.8])
    )
    world.scene.add(cable)
    
    print("Scene elements added!")
    
    # =========================================================================
    # INSTRUCTIONS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("SCENE READY")
    print("=" * 60)
    print("""
If UR5e didn't load automatically, you can load it manually:

1. In Isaac Sim, go to: Content Browser (bottom panel)
2. Navigate to: Isaac Sim > Robots > UniversalRobots
3. Drag 'ur5e.usd' into the viewport
4. Or use: Create > Isaac > Robots > UR5e

Camera controls:
- Right-click + drag: Rotate view
- Middle-click + drag: Pan
- Scroll: Zoom
- F: Frame selected object
    """)
    print("=" * 60)
    
    # Run simulation
    world.reset()
    
    print("\nRunning... Close window to exit.")
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()
    print("Done!")


if __name__ == "__main__":
    main()
