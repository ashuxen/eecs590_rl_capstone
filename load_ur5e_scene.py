"""
============================================================================
LOAD UR5e SCENE FROM USD for my project
============================================================================

Loads the saved UR5e cable insertion scene from USD file for testing.
This Scene was created using NVIDIA ISAAC sim abd saved as .usd file.

Run with Isaac Sim Python:
    C:\isaacsim\IsaacLab\_isaac_sim\python.bat load_ur5e_scene.py

Author: Ashutosh Kumar
============================================================================
"""

import os

def main():
    print("=" * 60)
    print("LOAD UR5e SCENE FROM USD")
    print("=" * 60)
    
    # Path to saved USD scene
    script_dir = os.path.dirname(os.path.abspath(__file__))
    usd_file = os.path.join(script_dir, "scenes", "ur5e_cable_insertion_scene.usd")
    
    if not os.path.exists(usd_file):
        print(f"\nERROR: Scene file not found: {usd_file}")
        print("Please save your scene first using Isaac Sim: File > Save As...")
        return
    
    print(f"\nLoading scene: {usd_file}")
    print()
    
    # Launch Isaac Sim
    print("Launching Isaac Sim...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})
    
    import omni.usd
    from omni.isaac.core import World
    
    # Open the saved USD scene
    print("Opening saved scene...")
    omni.usd.get_context().open_stage(usd_file)
    
    # Create world (without adding default ground plane - scene already has it)
    world = World(stage_units_in_meters=1.0)
    world.reset()
    
    print("\n" + "=" * 60)
    print("SCENE LOADED SUCCESSFULLY!")
    print("=" * 60)
    print(f"""
Scene: {usd_file}

Controls:
- Right Mouse + Drag: Rotate camera
- Middle Mouse + Drag: Pan camera  
- Scroll Wheel: Zoom
- W: Move tool
- E: Rotate tool
- R: Scale tool

Press PLAY button (or Space) to start simulation.
Close window when done.
    """)
    print("=" * 60)
    
    # Keep running until closed
    while simulation_app.is_running():
        world.step(render=True)
    
    simulation_app.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
