"""
================================================================================
EECS 590 Mini 2 Problem 1: Windy Chasm - Interactive Isaac Sim UI
================================================================================

Interactive visualization with UI controls:
- Sliders to adjust B, gamma
- Input fields for R_goal, r_crash, episodes
- Play button to run simulation
- Reset button to try again

Author: Ashutosh Kumar
================================================================================
"""

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Dict, List, Any

# =============================================================================
# MDP IMPLEMENTATION (from Mini 2 Solution)
# =============================================================================

@dataclass
class FiniteMDP:
    states: List[Any]
    actions: List[Any]
    P_a: Dict[Any, sp.csr_matrix]
    R_s: np.ndarray
    gamma: float

    def n_states(self):
        return len(self.states)

    def greedy_policy_from_V(self, V):
        n = self.n_states()
        pi = {}
        for s in range(n):
            best_a, best_q = None, -np.inf
            for a in self.actions:
                Pa = self.P_a[a].getrow(s)
                q_sa = float(self.R_s[s]) + self.gamma * float(Pa.dot(V).item())
                if q_sa > best_q:
                    best_q, best_a = q_sa, a
            pi[s] = {best_a: 1.0}
        return pi

    def value_iteration_control(self, tol=1e-10, max_iter=50000):
        n = self.n_states()
        V = np.zeros(n)
        for iteration in range(max_iter):
            V_new = np.empty_like(V)
            for s in range(n):
                q_vals = [float(self.R_s[s]) + self.gamma * float(self.P_a[a].getrow(s).dot(V).item()) 
                          for a in self.actions]
                V_new[s] = max(q_vals)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        return V, self.greedy_policy_from_V(V), iteration + 1


GRID_I, GRID_J = 20, 7
ACTIONS = ["F", "L", "R"]

def build_windy_chasm_mdp(B=0.5, R_goal=20.0, r_crash=5.0, gamma=0.99):
    states, state_to_idx = [], {}
    for i in range(GRID_I):
        for j in range(GRID_J):
            states.append((i, j))
            state_to_idx[(i, j)] = len(states) - 1

    crash_idx, goal_idx = len(states), len(states) + 1
    states.extend([("crash",), ("goal",)])
    n = len(states)

    def p_j(j):
        return B ** (1.0 / (1.0 + (j - 3) ** 2))

    def apply_wind(i, j):
        if j < 0 or j > 6:
            return {crash_idx: 1.0}
        if i >= GRID_I - 1:
            return {goal_idx: 1.0}
        
        pj = p_j(j)
        p1, p2 = pj, (1 - pj) * pj**2
        probs = {}
        
        def add(ii, jj, p):
            if p <= 0: return
            key = crash_idx if jj < 0 or jj > 6 else (goal_idx if ii >= GRID_I - 1 else state_to_idx[(ii, jj)])
            probs[key] = probs.get(key, 0) + p

        if j == 3:
            add(i, 2, 0.5 * p1); add(i, 4, 0.5 * p1)
            add(i, 1, 0.5 * p2); add(i, 5, 0.5 * p2)
            add(i, 3, 1 - p1 - p2)
        else:
            d = 1 if j > 3 else -1
            add(i, j + d, p1); add(i, j + 2*d, p2); add(i, j, 1 - p1 - p2)
        return probs

    P_a = {}
    for a in ACTIONS:
        rows, cols, data = [], [], []
        for (i, j), s_idx in state_to_idx.items():
            i2, j2 = (i+1, j) if a == "F" else ((i, j-1) if a == "L" else (i, j+1))
            for s2, p in apply_wind(i2, j2).items():
                rows.append(s_idx); cols.append(s2); data.append(p)
        for idx in (crash_idx, goal_idx):
            rows.append(idx); cols.append(idx); data.append(1.0)
        P_a[a] = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    R_s = np.full(n, -1.0)
    R_s[crash_idx], R_s[goal_idx] = -r_crash, R_goal
    return FiniteMDP(states, ACTIONS, P_a, R_s, gamma), state_to_idx, crash_idx, goal_idx


# =============================================================================
# ISAAC SIM WITH INTERACTIVE UI
# =============================================================================

def main():
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1920, "height": 1080})
    
    import omni.ui as ui
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualSphere
    
    # =========================================================================
    # SIMULATION STATE (using a class to hold mutable state)
    # =========================================================================
    class SimState:
        # Parameters (updated by UI)
        B = 0.5
        R_goal = 20.0
        r_crash = 5.0
        gamma = 0.99
        max_episodes = 5
        
        # Simulation state machine
        mode = "idle"  # "idle", "solving", "running", "paused"
        current_episode = 0
        current_step = 0
        current_state_idx = 0
        successes = 0
        crashes = 0
        frame_counter = 0
        
        # MDP data (computed when PLAY is pressed)
        mdp = None
        pi_star = None
        V_star = None
        state_to_idx = None
        crash_idx = None
        goal_idx = None
        
        # Random generator
        rng = np.random.default_rng()
    
    sim = SimState()
    
    # =========================================================================
    # CREATE WORLD AND OBJECTS
    # =========================================================================
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    
    # Create walls
    wall_left = FixedCuboid(
        prim_path="/World/WallLeft", name="wall_left", 
        position=np.array([10.0, -0.3, 0.5]),
        scale=np.array([21.0, 0.3, 1.5]),
        color=np.array([0.5, 0.25, 0.1])
    )
    world.scene.add(wall_left)
    
    wall_right = FixedCuboid(
        prim_path="/World/WallRight", name="wall_right",
        position=np.array([10.0, 7.3, 0.5]),
        scale=np.array([21.0, 0.3, 1.5]),
        color=np.array([0.5, 0.25, 0.1])
    )
    world.scene.add(wall_right)
    
    goal_zone = FixedCuboid(
        prim_path="/World/GoalZone", name="goal",
        position=np.array([19.5, 3.5, 0.01]),
        scale=np.array([1.0, 7.0, 0.02]),
        color=np.array([0.0, 0.8, 0.0])
    )
    world.scene.add(goal_zone)
    
    start_marker = FixedCuboid(
        prim_path="/World/StartMarker", name="start",
        position=np.array([0.5, 3.5, 0.01]),
        scale=np.array([1.0, 1.0, 0.02]),
        color=np.array([0.0, 0.0, 0.8])
    )
    world.scene.add(start_marker)
    
    center_line = FixedCuboid(
        prim_path="/World/CenterLine", name="center",
        position=np.array([10.0, 3.5, 0.005]),
        scale=np.array([20.0, 0.1, 0.01]),
        color=np.array([0.8, 0.8, 0.0])
    )
    world.scene.add(center_line)
    
    # =========================================================================
    # WIND STRENGTH VISUAL INDICATORS (bars at x=-1 showing p(j) per row)
    # =========================================================================
    wind_bars = []  # Store references to update later
    
    def create_wind_indicators(B_val):
        """Create/update wind strength bars at left side of chasm."""
        for j in range(7):
            # Compute p(j) for this row
            E_j = 1.0 / (1.0 + (j - 3) ** 2)
            pj = B_val ** E_j
            
            # Bar length proportional to wind probability (scale: 0-2 units)
            bar_length = pj * 2.0
            
            # Color: red (high wind) to blue (low wind)
            # High p(j) at center = more wind = more red
            red = pj
            blue = 1.0 - pj
            
            bar = FixedCuboid(
                prim_path=f"/World/WindBar_{j}",
                name=f"wind_bar_{j}",
                position=np.array([-1.5, j + 0.5, 0.3]),
                scale=np.array([bar_length, 0.6, 0.3]),
                color=np.array([red, 0.2, blue])
            )
            world.scene.add(bar)
            wind_bars.append(bar)
    
    # Create initial wind indicators with default B=0.5
    create_wind_indicators(0.5)
    
    # Wind indicator legend
    wind_legend = FixedCuboid(
        prim_path="/World/WindLegend",
        name="wind_legend",
        position=np.array([-2.5, 3.5, 0.1]),
        scale=np.array([0.5, 7.5, 0.1]),
        color=np.array([0.3, 0.3, 0.3])
    )
    world.scene.add(wind_legend)
    
    agent = VisualSphere(
        prim_path="/World/Agent", name="agent",
        position=np.array([0.5, 3.5, 0.4]),
        radius=0.35, color=np.array([1.0, 0.0, 0.0])
    )
    
    world.reset()
    
    # =========================================================================
    # HELPER: Compute p(j) for display
    # =========================================================================
    def compute_p_j(B, j):
        """Compute wind probability at position j given base B."""
        E_j = 1.0 / (1.0 + (j - 3) ** 2)
        return B ** E_j
    
    def get_wind_table_text(B):
        """Generate wind probability table for all j positions."""
        lines = ["Wind p(j) by position:"]
        for j in range(7):
            pj = compute_p_j(B, j)
            bar_len = int(pj * 20)  # Scale to 20 chars
            bar = "#" * bar_len + "-" * (20 - bar_len)
            lines.append(f"j={j}: {pj:.3f} |{bar}|")
        return "\n".join(lines)
    
    # =========================================================================
    # UI WINDOW
    # =========================================================================
    window = ui.Window("Mini 2: Windy Chasm Controls", width=450, height=700)
    
    # UI references
    status_label = None
    value_label = None
    results_label = None
    wind_current_label = None
    wind_table_label = None
    
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("EECS 590 Mini 2 Problem 1", style={"font_size": 18, "color": 0xFFFFFF00})
            ui.Label("Windy Chasm MDP Visualization", style={"font_size": 14})
            ui.Spacer(height=5)
            
            # ===== B slider =====
            with ui.HStack(height=25):
                ui.Label("B (Wind Prob):", width=130)
                B_slider = ui.FloatSlider(min=0.1, max=0.9, step=0.05)
                B_slider.model.set_value(0.5)
                B_val_label = ui.Label("0.50", width=50)
            
            # ===== Wind probability table (updates when B changes) =====
            ui.Spacer(height=5)
            ui.Label("p(j) = B^(1/(1+(j-3)^2))", style={"font_size": 11, "color": 0xFFAAAAAA})
            wind_table_label = ui.Label(get_wind_table_text(0.5), style={"font_size": 10})
            
            def on_B_changed(model):
                sim.B = model.as_float
                B_val_label.text = f"{sim.B:.2f}"
                # Update wind table when B changes
                wind_table_label.text = get_wind_table_text(sim.B)
            B_slider.model.add_value_changed_fn(on_B_changed)
            
            ui.Spacer(height=5)
            
            # ===== REAL-TIME Wind at current position =====
            with ui.HStack(height=30):
                ui.Label("Current Wind:", width=100)
                wind_current_label = ui.Label("p(3) = 0.500 [at start]", 
                                              style={"font_size": 14, "color": 0xFFFF6600})
            
            ui.Spacer(height=5)
            
            # ===== Gamma slider =====
            with ui.HStack(height=25):
                ui.Label("Gamma:", width=130)
                gamma_slider = ui.FloatSlider(min=0.8, max=0.999, step=0.01)
                gamma_slider.model.set_value(0.99)
                gamma_val_label = ui.Label("0.99", width=50)
            
            def on_gamma_changed(model):
                sim.gamma = model.as_float
                gamma_val_label.text = f"{sim.gamma:.3f}"
            gamma_slider.model.add_value_changed_fn(on_gamma_changed)
            
            # ===== R_goal input =====
            with ui.HStack(height=25):
                ui.Label("R_goal:", width=130)
                R_goal_field = ui.FloatField(width=80)
                R_goal_field.model.set_value(20.0)
            
            def on_R_goal_changed(model):
                sim.R_goal = model.as_float
            R_goal_field.model.add_value_changed_fn(on_R_goal_changed)
            
            # ===== r_crash input =====
            with ui.HStack(height=25):
                ui.Label("r_crash:", width=130)
                r_crash_field = ui.FloatField(width=80)
                r_crash_field.model.set_value(5.0)
            
            def on_r_crash_changed(model):
                sim.r_crash = model.as_float
            r_crash_field.model.add_value_changed_fn(on_r_crash_changed)
            
            # ===== Episodes input =====
            with ui.HStack(height=25):
                ui.Label("Episodes:", width=130)
                episodes_field = ui.IntField(width=80)
                episodes_field.model.set_value(5)
            
            def on_episodes_changed(model):
                sim.max_episodes = model.as_int
            episodes_field.model.add_value_changed_fn(on_episodes_changed)
            
            ui.Spacer(height=10)
            
            # ===== Value display =====
            value_label = ui.Label("V*(0,3) = ---", style={"font_size": 16, "color": 0xFF00FF00})
            
            ui.Spacer(height=5)
            
            # ===== Buttons =====
            with ui.HStack(height=40, spacing=10):
                play_btn = ui.Button("PLAY", width=150, height=40)
                play_btn.set_style({"background_color": 0xFF228B22})
                
                reset_btn = ui.Button("RESET", width=100, height=40)
                reset_btn.set_style({"background_color": 0xFF8B4513})
            
            ui.Spacer(height=10)
            
            # ===== Status and results =====
            status_label = ui.Label("Ready. Adjust parameters and click PLAY.", style={"font_size": 12})
            results_label = ui.Label("", style={"font_size": 14, "color": 0xFFFFFFFF})
    
    # =========================================================================
    # BUTTON CALLBACKS
    # =========================================================================
    def on_play_clicked():
        if sim.mode != "idle":
            return
        
        # Reset counters
        sim.successes = 0
        sim.crashes = 0
        sim.current_episode = 0
        sim.current_step = 0
        sim.frame_counter = 0
        
        # Print wind probability table for current B
        print("\n" + "=" * 60)
        print(f"STARTING SIMULATION with B = {sim.B:.2f}")
        print("=" * 60)
        print(f"Wind probabilities p(j) = {sim.B:.2f}^(1/(1+(j-3)^2)):")
        print("-" * 60)
        for j in range(7):
            E_j = 1.0 / (1.0 + (j - 3) ** 2)
            pj = sim.B ** E_j
            bar = "#" * int(pj * 30) + "-" * (30 - int(pj * 30))
            wind_dir = "CENTER (both)" if j == 3 else ("toward j=6" if j < 3 else "toward j=0")
            print(f"  j={j}: p={pj:.3f} |{bar}| Wind pushes {wind_dir}")
        print("-" * 60)
        
        # Build MDP
        status_label.text = f"Building MDP (B={sim.B:.2f})..."
        sim.mdp, sim.state_to_idx, sim.crash_idx, sim.goal_idx = build_windy_chasm_mdp(
            B=sim.B, R_goal=sim.R_goal, r_crash=sim.r_crash, gamma=sim.gamma
        )
        
        # Solve MDP
        status_label.text = "Running Value Iteration..."
        sim.V_star, sim.pi_star, iters = sim.mdp.value_iteration_control()
        
        v_start = sim.V_star[sim.state_to_idx[(0, 3)]]
        value_label.text = f"V*(0,3) = {v_start:.2f}"
        
        print(f"\nValue Iteration converged in {iters} iterations")
        print(f"V*(0,3) = {v_start:.4f}")
        print("=" * 60 + "\n")
        
        status_label.text = f"Converged in {iters} iters. Starting simulation..."
        results_label.text = ""
        
        # Start first episode
        sim.current_state_idx = sim.state_to_idx[(0, 3)]
        agent.set_world_pose(position=np.array([0.5, 3.5, 0.4]))
        wind_current_label.text = f"p(3) = {compute_p_j(sim.B, 3):.3f} | Wind: CENTER"
        
        sim.mode = "running"
    
    def on_reset_clicked():
        sim.mode = "idle"
        sim.current_episode = 0
        sim.successes = 0
        sim.crashes = 0
        agent.set_world_pose(position=np.array([0.5, 3.5, 0.4]))
        status_label.text = "Reset. Ready to play."
        value_label.text = "V*(0,3) = ---"
        results_label.text = ""
    
    play_btn.set_clicked_fn(on_play_clicked)
    reset_btn.set_clicked_fn(on_reset_clicked)
    
    # =========================================================================
    # MAIN SIMULATION LOOP
    # =========================================================================
    print("=" * 70)
    print("Mini 2: Windy Chasm - Interactive UI")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("WIND PROBABILITY FORMULA:")
    print("  p(j) = B^E(j)  where  E(j) = 1 / (1 + (j-3)^2)")
    print("=" * 70)
    print("\nHow B affects wind at each row (for B=0.5):")
    print("-" * 50)
    for j in range(7):
        E_j = 1.0 / (1.0 + (j - 3) ** 2)
        pj = 0.5 ** E_j
        dist = abs(j - 3)
        bar = "#" * int(pj * 30) + "-" * (30 - int(pj * 30))
        print(f"  j={j} (dist={dist}): E={E_j:.3f}, p={pj:.3f} |{bar}|")
    print("-" * 50)
    print("\nNOTE: p(j) is HIGHEST at center (j=3) where wind is STRONGEST")
    print("      p(j) DECREASES toward walls where wind is WEAKER")
    print("\n3D SCENE: Red bars on left show wind strength per row")
    print("          Longer/redder = stronger wind")
    print("\nUse the control panel to adjust B and see how wind changes!")
    print("Close the window to exit.\n")
    
    FRAMES_PER_STEP = 15  # Animation speed
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if sim.mode == "running":
            sim.frame_counter += 1
            
            if sim.frame_counter >= FRAMES_PER_STEP:
                sim.frame_counter = 0
                
                # Check terminal states
                if sim.current_state_idx == sim.crash_idx:
                    sim.crashes += 1
                    status_label.text = f"Episode {sim.current_episode + 1}: CRASHED!"
                    sim.current_episode += 1
                    
                    if sim.current_episode >= sim.max_episodes:
                        sim.mode = "idle"
                        results_label.text = f"Done! Success: {sim.successes}/{sim.max_episodes}, Crashes: {sim.crashes}/{sim.max_episodes}"
                        status_label.text = "Simulation complete."
                    else:
                        # Reset for next episode
                        sim.current_state_idx = sim.state_to_idx[(0, 3)]
                        sim.current_step = 0
                        agent.set_world_pose(position=np.array([0.5, 3.5, 0.4]))
                    continue
                
                elif sim.current_state_idx == sim.goal_idx:
                    sim.successes += 1
                    status_label.text = f"Episode {sim.current_episode + 1}: SUCCESS!"
                    sim.current_episode += 1
                    
                    if sim.current_episode >= sim.max_episodes:
                        sim.mode = "idle"
                        results_label.text = f"Done! Success: {sim.successes}/{sim.max_episodes}, Crashes: {sim.crashes}/{sim.max_episodes}"
                        status_label.text = "Simulation complete."
                    else:
                        # Reset for next episode
                        sim.current_state_idx = sim.state_to_idx[(0, 3)]
                        sim.current_step = 0
                        agent.set_world_pose(position=np.array([0.5, 3.5, 0.4]))
                    continue
                
                # Get current state position
                s = sim.mdp.states[sim.current_state_idx]
                if isinstance(s, tuple) and len(s) == 2:
                    i, j = s
                    pos = np.array([i + 0.5, j + 0.5, 0.4])
                    agent.set_world_pose(position=pos)
                    
                    # ===== UPDATE REAL-TIME WIND PROBABILITY DISPLAY =====
                    current_pj = compute_p_j(sim.B, j)
                    wind_dir = "CENTER" if j == 3 else ("UP" if j < 3 else "DOWN")
                    wind_current_label.text = f"p({j}) = {current_pj:.3f} | Wind: {wind_dir}"
                    
                    # Get action from optimal policy
                    action = list(sim.pi_star[sim.current_state_idx].keys())[0]
                    
                    # Sample next state stochastically
                    P = sim.mdp.P_a[action].getrow(sim.current_state_idx)
                    sim.current_state_idx = sim.rng.choice(P.indices, p=P.data)
                    
                    sim.current_step += 1
                    status_label.text = f"Ep {sim.current_episode + 1}/{sim.max_episodes}, Step {sim.current_step}: ({i},{j}) -> {action}"
                    
                    # Safety limit
                    if sim.current_step > 100:
                        sim.current_episode += 1
                        if sim.current_episode >= sim.max_episodes:
                            sim.mode = "idle"
                            results_label.text = f"Done! Success: {sim.successes}/{sim.max_episodes}, Crashes: {sim.crashes}/{sim.max_episodes}"
                        else:
                            sim.current_state_idx = sim.state_to_idx[(0, 3)]
                            sim.current_step = 0
                            agent.set_world_pose(position=np.array([0.5, 3.5, 0.4]))
    
    simulation_app.close()


if __name__ == "__main__":
    main()
