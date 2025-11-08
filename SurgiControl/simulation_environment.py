#!/usr/bin/env python3
# /// script
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pygame",
# ]
# ///

"""
Serial-chain 7-DOF demo robot with classic DH forward kinematics and 3D animation.

Key points
----------
- Classic DH: each link i has (a_i, alpha_i, d_i, theta_i)
- FK composes T = T_01 * T_12 * ... * T_{n-1,n}
- Demo IK: base yaw + 2-link planar shoulder/elbow to reach (x,y,z) approximately.
  The wrist joints default to zero to keep the example simple.

Angles are in radians.
"""

from typing import Optional, Sequence, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import time


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Classic DH homogeneous transform."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ], dtype=float)


class SerialRobotSim:
    def __init__(
        self,
        a_list: Optional[Sequence[float]] = None,
        alpha_list: Optional[Sequence[float]] = None,
        d_list: Optional[Sequence[float]] = None,
        dof: int = 7,
    ):
        """
        Construct a serial-chain robot using classic DH params.

        Defaults are a simple, non-degenerate 7-link chain:
        - a: link lengths (roughly like your original)
        - alpha: alternating twists to create a spatial chain
        - d: zero offsets (all-revolute about each link's z-axis)
        """
        if a_list is None:
            a_list = [7, 6, 4, 5, 3, 2, 1.5]
        if alpha_list is None:
            # A simple alternating pattern to avoid colinear axes
            alpha_list = [np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2,
                          np.pi / 2, 0]
        if d_list is None:
            d_list = [0.0] * dof

        self.a: List[float] = list(a_list)
        self.alpha: List[float] = list(alpha_list)
        self.d: List[float] = list(d_list)

        if not (len(self.a) == len(self.alpha) == len(self.d) == dof):
            raise ValueError("DH parameter lists must all have length == dof")

        self.dof = dof

        self.fig = None
        self.ax = None
        self.joint_lines = None
        self.trajectory_line = None

        # Animation data
        self.joint_angles_time_series: Optional[List[List[float]]] = None
        self.ee_traj: List[Tuple[float, float, float]] = []

    # -------------------------
    # Forward kinematics (FK)
    # -------------------------
    def forward_kinematics(self, joint_angles: Sequence[float]):
        """
        Compute cumulative transforms and joint origins.

        Returns:
            positions: list of 3D points for joint i origins in world frame,
                       including the end-effector origin as the last element.
            transforms: list of 4x4 cumulative transforms T_0i
        """
        if len(joint_angles) != self.dof:
            raise ValueError("Expected %d joint angles" % self.dof)

        T = np.eye(4)
        positions = [(0.0, 0.0, 0.0)]
        transforms = [T.copy()]

        for i in range(self.dof):
            Ti = dh_transform(self.a[i], self.alpha[i], self.d[i],
                              joint_angles[i])
            T = T @ Ti
            positions.append(tuple(T[:3, 3]))
            transforms.append(T.copy())

        # positions has length dof+1 (base + each joint/end)
        return positions, transforms

    # -------------------------
    # Demo inverse kinematics
    # -------------------------
    def demo_inverse_kinematics(self, x: float, y: float, z: float) -> List[
        float]:
        """
        Toy IK for demonstration (not a general solver):
        - q0 (base yaw) aims the arm toward (x, y)
        - q1, q2 solve a planar 2-link problem in the r-z plane (r = horizontal reach)
        - remaining joints are set to 0 to form a neutral wrist

        It uses the first two link lengths (a0, a1) for the planar reach.
        """
        q = [0.0] * self.dof

        # Base yaw
        q[0] = np.arctan2(y, x)

        # Project target into the base-aligned plane
        r = np.hypot(x, y)
        zt = z

        L1 = abs(self.a[1]) if self.dof > 1 else 0.0
        L0 = abs(self.a[0]) if self.dof > 0 else 0.0

        # Clamp to reachable workspace of the 2-link subset
        R = np.hypot(r, zt)        
        R = max(min(R, L0 + L1 - 1e-9), 1e-9)

        # Law of cosines for elbow
        cos_elbow = (R ** 2 - L0 ** 2 - L1 ** 2) / (2 * L0 * L1)
        cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
        elbow = np.arccos(cos_elbow)  # choose "elbow-down" (0..pi)

        # Shoulder from triangle geometry
        gamma = np.arctan2(zt, r)
        phi = np.arctan2(L1 * np.sin(elbow), L0 + L1 * np.cos(elbow))
        shoulder = gamma - phi

        # Map to q1, q2 (assumes revolute joints about z with DH twists providing plane)
        q[1] = shoulder
        q[2] = elbow

        # Wrist neutral
        # q[3:] remain 0.0

        return q

    def setup_axes_3d(self, title: Optional[str] = None, auto_rotate: bool = False):
        reach = sum(abs(a) for a in self.a) + sum(abs(d) for d in self.d)
        lim = (-reach, reach) if reach > 0 else (-10, 10)

        self.fig = plt.figure(figsize=(16, 9), dpi=80)
        self.ax = self.fig.add_subplot(111, projection='3d')
        plt.ion()  # enable interactive mode

        self.ax.set_xlim(*lim)
        self.ax.set_ylim(*lim)
        self.ax.set_zlim(*lim)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        if title:
            self.ax.set_title(title)

        # Lines for robot links
        self.joint_lines = [
            self.ax.plot([], [], [], 'o-', linewidth=2, markersize=4)[0]
            for _ in range(self.dof)
        ]

        # Line for end-effector trajectory (toggle visibility later)
        (self.trajectory_line,) = self.ax.plot([], [], [], 'r--', linewidth=1.5, visible=True)

        # Control flag: show/hide trajectory
        self.show_trajectory = True
        self.auto_rotate = auto_rotate  # Enable/disable auto rotation

    def draw_pose(self, joint_angles: Sequence[float]):
        positions, _ = self.forward_kinematics(joint_angles)

        # Draw links
        for j, line in enumerate(self.joint_lines):
            a = positions[j]
            b = positions[j + 1]
            xs, ys, zs = zip(a, b)
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        # Update trajectory
        ee = positions[-1]
        if self.show_trajectory:
            self.ee_traj.append(ee)
            if len(self.ee_traj) > 1:
                tx, ty, tz = (np.array(t) for t in zip(*self.ee_traj))
                self.trajectory_line.set_data(tx, ty)
                self.trajectory_line.set_3d_properties(tz)
                self.trajectory_line.set_visible(True)
        else:
            self.trajectory_line.set_visible(False)

        # Optional: auto rotate (only if enabled)
        if hasattr(self, "auto_rotate") and self.auto_rotate:
            if not hasattr(self, "azim_angle"):
                self.azim_angle = 0
            self.azim_angle += 1
            self.ax.view_init(elev=30, azim=self.azim_angle)

        return self.joint_lines + [self.trajectory_line]


    def animate_3d(self, frame: int):
        if self.joint_angles_time_series is None or frame >= len(
            self.joint_angles_time_series):
            return []
        return self.draw_pose(self.joint_angles_time_series[frame])

    # -------------------------
    # Trajectories / utilities
    # -------------------------
    @staticmethod
    def create_time_series(start: float, end: float, step: float):
        """Inclusive end for convenience in demos."""
        return np.arange(start, end + step, step)


def simple_static_pose_demo():
    """
    Demo of plotting a static pose from a joint command with the robot.
    """
    sim = SerialRobotSim()
    sim.setup_axes_3d(title="simple_static_pose_demo")
    q_static = [-1.0, 0.6, 0.8, -0.3, 0.9, -0.5, 0.2]
    sim.draw_pose(q_static)
    plt.ioff()  # Disable interactive mode for blocking display
    plt.show(block=True)
    plt.ion()  # Re-enable interactive mode


def simple_joint_trajectory_demo():
    """
    Demo of passing in a series of joint commands to the robot and animating
    the result.
    """
    sim = SerialRobotSim()

    # Time grid
    t0, t_end, dt = 0.0, 9.0, 0.02
    T = sim.create_time_series(t0, t_end, dt)

    # Waypoints: (time, q) with q shape == (dof,)
    q_rest = np.zeros(sim.dof)
    q_pose1 = np.array([-0.6, 0.8, 0.9, -0.4, 0.6, -0.3, 0.2])
    q_pose2 = np.array([0.8, -0.5, 0.7, 0.6, -0.4, 0.3, -0.2])
    q_pose3 = np.array([0.0, 0.3, -0.6, 0.4, 0.2, -0.4, 0.5])

    waypoints = [
        (0.0, q_rest),
        (3.0, q_pose1),
        (6.0, q_pose2),
        (9.0, q_pose3),
    ]

    # Stack waypoint times and values
    t_wp = np.array([t for t, _ in waypoints], dtype=float)
    q_wp = np.vstack([q for _, q in waypoints])  # shape: (num_wp, dof)

    # Linear interpolation per joint
    Q = np.column_stack(
        [np.interp(T, t_wp, q_wp[:, j]) for j in range(sim.dof)])

    # Animate with these joint angles
    sim.joint_angles_time_series = [q.tolist() for q in Q]
    sim.ee_traj = []

    sim.setup_axes_3d(title="simple_joint_trajectory_demo", auto_rotate=True)
    anim = animation.FuncAnimation(
        sim.fig, sim.animate_3d,
        frames=len(T),
        interval=25,
        blit=False,
        repeat=False
    )
    sim.anim = anim  # Store reference to keep animation alive

    plt.ioff()  # Disable interactive mode for blocking display
    plt.show(block=True)
    plt.ion()  # Re-enable interactive mode
    return anim


def simple_ik_trajectory_demo():
    """
    This is really intended to just be a demo to get you started. Highly
    encouraged to implement your own IK and/or use an existing library, as well
    as come up with your own trajectories and motion planning.
    """

    sim = SerialRobotSim()

    t0, t_end, dt = 0.0, 10.0, 0.02  # ~500 frames, smooth & fast
    time_series = sim.create_time_series(t0, t_end, dt)

    def figure_eight_trajectory_3d(t: float, scale: float = 5.0):
        """3D figure-eight trajectory for the end-effector."""
        x = scale * np.sin(2 * np.pi * t)
        y = scale * np.sin(2 * np.pi * t) * np.cos(2 * np.pi * t)
        z = scale * np.cos(4 * np.pi * t)
        return x, y, z

    joint_angles_time_series: List[List[float]] = []
    for t in time_series:
        x, y, z = figure_eight_trajectory_3d(t, scale=8.0)
        q = sim.demo_inverse_kinematics(x, y, z)
        joint_angles_time_series.append(q)

    sim.joint_angles_time_series = joint_angles_time_series
    sim.ee_traj = []  # reset trajectory for the new animation

    sim.setup_axes_3d(title="simple_ik_trajectory_demo", auto_rotate=True)
    anim = animation.FuncAnimation(
        sim.fig, sim.animate_3d,
        frames=len(time_series),
        interval=50,
        blit=False,
        repeat=False
    )
    sim.anim = anim  # Store reference to keep animation alive

    plt.ioff()  # Disable interactive mode for blocking display
    plt.show(block=True)
    plt.ion()  # Re-enable interactive mode
    return anim


def simple_control_loop_demo():
    """
    Tick-based joint-trajectory demo:
    - Build a linear joint-space trajectory from waypoints
    - At fixed ticks (50 ms), compute q_cmd(t) and "apply" it
    - Optionally simulate 1-tick latency (sample-and-hold) for q_act
    - Redraw robot each tick and plot inputs/outputs afterward
    """
    import time

    sim = SerialRobotSim()
    sim.setup_axes_3d(title="simple_control_loop_demo")
    sim.ee_traj = []

    # trajectory definition
    t0, t_end, dt_traj = 0.0, 9.0, 0.02
    T_traj = sim.create_time_series(t0, t_end, dt_traj)

    q_rest = np.zeros(sim.dof)
    q_pose1 = np.array([-0.6, 0.8, 0.9, -0.4, 0.6, -0.3, 0.2])
    q_pose2 = np.array([ 0.8,-0.5, 0.7, 0.6,-0.4, 0.3,-0.2])
    q_pose3 = np.array([ 0.0, 0.3,-0.6, 0.4, 0.2,-0.4, 0.5])

    waypoints = [
        (0.0, q_rest),
        (3.0, q_pose1),
        (6.0, q_pose2),
        (9.0, q_pose3),
    ]
    t_wp = np.array([t for t, _ in waypoints], dtype=float)
    q_wp = np.vstack([q for _, q in waypoints])  # (num_wp, dof)

    # Precompute a simple linear trajectory q_ref(t) on T_traj for convenience
    Q_ref = np.column_stack([np.interp(T_traj, t_wp, q_wp[:, j]) for j in range(sim.dof)])

    # Helper to grab q_ref for any time t via interpolation on the precomputed arrays
    def q_ref_at(t: float) -> np.ndarray:
        t_clamped = np.clip(t, T_traj[0], T_traj[-1])
        # Find index and linear interpolate between neighboring samples
        idx = np.searchsorted(T_traj, t_clamped)
        if idx == 0:
            return Q_ref[0].copy()
        if idx >= len(T_traj):
            return Q_ref[-1].copy()
        t0, t1 = T_traj[idx-1], T_traj[idx]
        a = (t_clamped - t0) / (t1 - t0)
        return (1 - a) * Q_ref[idx-1] + a * Q_ref[idx]

    # tick loop
    dt_tick = 0.050 # 50 ms tick, 20 Hz
    latency_ticks = 1 # set to 0 for no latency; 1 to simulate 1-tick delay

    # Simple FIFO to model actuator latency
    cmd_fifo: List[np.ndarray] = []

    # telemetry log of timestamps
    t_log: List[float] = []

    # telemetry log of commanded joint position at time t
    q_cmd_log: List[np.ndarray] = []
    # telemetry log of actual joint position at time t
    q_act_log: List[np.ndarray] = []

    # Timing
    t_start = time.perf_counter()
    next_tick = t_start

    # Run until a small hold after t_end
    while True:
        if sim.fig is None or not plt.fignum_exists(sim.fig.number):
            break

        now = time.perf_counter()
        t = now - t_start
        if t >= t_end + 0.25:
            break

        # Input: commanded joints from trajectory
        q_cmd = q_ref_at(t)

        # Push into FIFO; pop for actual applied joints
        cmd_fifo.append(q_cmd.copy())
        if len(cmd_fifo) > latency_ticks:
            q_act = cmd_fifo.pop(0)
        else:
            q_act = q_cmd  # not enough items yet; act immediately

        # Draw applied joints
        sim.draw_pose(q_act)
        plt.pause(0.001)

        # Log
        t_log.append(t)
        q_cmd_log.append(q_cmd.copy())
        q_act_log.append(q_act.copy())

        # Sleep to next tick
        next_tick += dt_tick
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # Overrun: resync to avoid drift
            next_tick = time.perf_counter()

    # quick visualization of input vs output for one joint J1
    t_arr = np.array(t_log)
    q_cmd_arr = np.vstack(q_cmd_log)
    q_act_arr = np.vstack(q_act_log)

    plt.figure(figsize=(10, 4))
    plt.plot(t_arr, q_cmd_arr[:, 0], label="cmd J1")
    plt.plot(t_arr, q_act_arr[:, 0], label="act J1", linestyle="--")
    plt.title("simple_control_loop_demo: commanded vs applied (Joint 1)")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [rad]")
    plt.legend()
    plt.grid(True)

    plt.ioff()  # Disable interactive mode for blocking display
    plt.show(block=True)
    plt.ion()  # Re-enable interactive mode

def interactive_input_demo():
    """
    Interactive demo where you can input joint angles to move the robot.
    Supports both keyboard control and command-line input.
    """
    sim = SerialRobotSim()
    sim.setup_axes_3d(title="Interactive Robot Control - Close window to exit")
    sim.ee_traj = []
    
    # Initial joint angles (all zeros = rest position)
    joint_angles = [0.0] * sim.dof
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3", "Gripper"]
    
    print("\n" + "="*60)
    print("Interactive Robot Control")
    print("="*60)
    print("\nCommands:")
    print("  - Enter joint angles: 'j1 j2 j3 j4 j5 j6 j7' (in radians)")
    print("  - Move single joint: 'j1 0.5' (joint_index angle)")
    print("  - Reset to zero: 'reset'")
    print("  - Show current: 'show'")
    print("  - IK mode: 'ik x y z' (target end-effector position)")
    print("  - Toggle trajectory: 'trajectory on/off'")
    print("  - Quit: 'quit' or 'exit'")
    print("\nKeyboard Controls (when plot window is focused):")
    print("  - Arrow keys: Adjust base joint (Left/Right)")
    print("  - WASD: Adjust shoulder/elbow (W/S: shoulder, A/D: elbow)")
    print("  - QE: Adjust wrist (Q/E)")
    print("  - R: Reset all joints to zero")
    print("  - T: Toggle trajectory display")
    print("="*60 + "\n")
    
    # Draw initial pose
    sim.draw_pose(joint_angles)
    plt.draw()
    plt.pause(0.1)
    
    # Keyboard event handler
    def on_key(event):
        nonlocal joint_angles
        step = 0.1  # radians per keypress
        
        if event.key == 'left':
            joint_angles[0] -= step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 'right':
            joint_angles[0] += step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 'w':
            joint_angles[1] += step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 's':
            joint_angles[1] -= step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 'a':
            joint_angles[2] -= step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 'd':
            joint_angles[2] += step
            sim.draw_pose(joint_angles)
            plt.draw()
        elif event.key == 'q':
            if sim.dof > 3:
                joint_angles[3] -= step
                sim.draw_pose(joint_angles)
                plt.draw()
        elif event.key == 'e':
            if sim.dof > 3:
                joint_angles[3] += step
                sim.draw_pose(joint_angles)
                plt.draw()
        elif event.key == 'r':
            joint_angles = [0.0] * sim.dof
            sim.ee_traj = []
            sim.draw_pose(joint_angles)
            plt.draw()
            print("Reset to zero position")
        elif event.key == 't':
            sim.show_trajectory = not sim.show_trajectory
            sim.draw_pose(joint_angles)
            plt.draw()
            print(f"Trajectory: {'ON' if sim.show_trajectory else 'OFF'}")
    
    # Connect keyboard event
    sim.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Keep window responsive while processing input
    print("Robot is ready! Enter commands in the terminal or use keyboard in the plot window.")
    print("Current joint angles:", [f"{q:.3f}" for q in joint_angles])
    
    try:
        while plt.fignum_exists(sim.fig.number):
            # Process input commands (non-blocking check)
            plt.pause(0.1)  # Allow plot to update
            
            # For command-line input, we'll use a thread or just prompt
            # Since matplotlib blocks, we'll use a simpler approach:
            # Show the window and wait for it to close, keyboard input works via events
            
    except KeyboardInterrupt:
        print("\nExiting...")
    
    # Alternative: Command-line input mode (uncomment to use)
    # Uncomment the section below if you prefer command-line input over keyboard


def interactive_command_line_demo():
    """
    Command-line input mode where you type joint angles to control the robot.
    """
    sim = SerialRobotSim()
    sim.setup_axes_3d(title="Interactive Robot Control (Command Line)")
    sim.ee_traj = []
    
    joint_angles = [0.0] * sim.dof
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3", "Gripper"]
    
    print("\n" + "="*60)
    print("Interactive Robot Control - Command Line Mode")
    print("="*60)
    print("\nEnter joint angles (in radians) or commands:")
    print("  Examples:")
    print("    '0.5 0.3 -0.2 0.0 0.0 0.0 0.0' - Set all 7 joints")
    print("    'j1 0.5' - Set joint 1 to 0.5 radians")
    print("    'reset' - Reset all joints to zero")
    print("    'ik 5 3 2' - Use IK to reach position (x=5, y=3, z=2)")
    print("    'show' - Show current joint angles")
    print("    'trajectory off' - Hide trajectory")
    print("    'quit' - Exit")
    print("="*60 + "\n")
    
    sim.draw_pose(joint_angles)
    plt.ioff()
    plt.show(block=False)
    
    while True:
        try:
            cmd = input("\nEnter command (or 'quit' to exit): ").strip().lower()
            
            if cmd in ['quit', 'exit', 'q']:
                break
            elif cmd == 'reset':
                joint_angles = [0.0] * sim.dof
                sim.ee_traj = []
                print("Reset to zero position")
            elif cmd == 'show':
                print(f"\nCurrent joint angles (radians):")
                for i, (name, angle) in enumerate(zip(joint_names[:sim.dof], joint_angles)):
                    print(f"  {name} (J{i}): {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
                positions, _ = sim.forward_kinematics(joint_angles)
                ee_pos = positions[-1]
                print(f"\nEnd-effector position: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
                continue
            elif cmd.startswith('ik '):
                try:
                    parts = cmd.split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    joint_angles = sim.demo_inverse_kinematics(x, y, z)
                    print(f"IK solution to reach ({x}, {y}, {z}):")
                    for i, angle in enumerate(joint_angles):
                        print(f"  J{i}: {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
                except (ValueError, IndexError):
                    print("Error: Invalid IK command. Use 'ik x y z'")
                    continue
            elif cmd.startswith('trajectory '):
                if 'off' in cmd:
                    sim.show_trajectory = False
                    print("Trajectory display: OFF")
                elif 'on' in cmd:
                    sim.show_trajectory = True
                    print("Trajectory display: ON")
                else:
                    print("Usage: 'trajectory on' or 'trajectory off'")
                    continue
            elif cmd.startswith('j'):
                # Single joint command: "j1 0.5"
                try:
                    parts = cmd.split()
                    joint_idx = int(parts[0][1:]) - 1  # j1 -> index 0
                    angle = float(parts[1])
                    if 0 <= joint_idx < sim.dof:
                        joint_angles[joint_idx] = angle
                        print(f"Set {joint_names[joint_idx]} (J{joint_idx+1}) to {angle:.4f} rad")
                    else:
                        print(f"Error: Joint index must be between 1 and {sim.dof}")
                        continue
                except (ValueError, IndexError):
                    print("Error: Invalid joint command. Use 'j1 0.5' format")
                    continue
            else:
                # Try to parse as space-separated joint angles
                try:
                    angles = [float(x) for x in cmd.split()]
                    if len(angles) == sim.dof:
                        joint_angles = angles
                        print(f"Set all joints: {[f'{a:.4f}' for a in joint_angles]}")
                    elif len(angles) < sim.dof:
                        # Update only the first N joints
                        for i, angle in enumerate(angles):
                            if i < sim.dof:
                                joint_angles[i] = angle
                        print(f"Updated first {len(angles)} joints")
                    else:
                        print(f"Error: Expected {sim.dof} joint angles, got {len(angles)}")
                        continue
                except ValueError:
                    print(f"Error: Could not parse command. Type 'quit' to exit.")
                    continue
            
            # Update visualization
            sim.draw_pose(joint_angles)
            plt.draw()
            plt.pause(0.01)
            
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
    
    plt.close(sim.fig)
    print("Goodbye!")


def interactive_controller_demo():
    """
    Interactive demo using a DualSense (or other game controller) to control the robot.
    Controller mapping:
    - Left Stick X: Base rotation (left/right)
    - Left Stick Y: Shoulder joint (up/down)
    - Right Stick X: Elbow joint (left/right)
    - Right Stick Y: Wrist joint (up/down)
    - L1/R1: Adjust wrist joints 2 and 3
    - L2/R2: Fine control for selected joint
    - Triangle: Reset to zero position
    - Circle: Toggle trajectory display
    - Square: Toggle trajectory recording
    - Cross: Exit
    """
    # Initialize pygame for controller input
    try:
        pygame.init()
        if not pygame.get_init():
            print("ERROR: Failed to initialize pygame!")
            print("Try reinstalling pygame: pip install --upgrade pygame")
            return
    except Exception as e:
        print(f"ERROR: Failed to initialize pygame: {e}")
        print("Try reinstalling pygame: pip install --upgrade pygame")
        return
    
    try:
        pygame.joystick.init()
        if not pygame.joystick.get_init():
            print("ERROR: Failed to initialize joystick subsystem!")
            return
    except Exception as e:
        print(f"ERROR: Failed to initialize joystick subsystem: {e}")
        return
    
    # Check for connected controllers
    try:
        joystick_count = pygame.joystick.get_count()
    except Exception as e:
        print(f"ERROR: Failed to get joystick count: {e}")
        pygame.quit()
        return
    
    if joystick_count == 0:
        print("ERROR: No controller detected!")
        print("Please connect your DualSense controller and try again.")
        print("\nTroubleshooting:")
        print("  1. Make sure controller is connected via USB or Bluetooth")
        print("  2. On Windows: Check Device Manager for controller")
        print("  3. Try running: python test_controller.py")
        pygame.quit()
        return
    
    # Initialize the first controller
    try:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    except Exception as e:
        print(f"ERROR: Failed to initialize controller: {e}")
        pygame.quit()
        return
    
    try:
        controller_name = joystick.get_name()
        num_axes = joystick.get_numaxes()
        num_buttons = joystick.get_numbuttons()
    except Exception as e:
        print(f"ERROR: Failed to get controller info: {e}")
        pygame.quit()
        return
    
    print(f"\n{'='*60}")
    print(f"Controller detected: {controller_name}")
    print(f"  Axes: {num_axes}, Buttons: {num_buttons}")
    print(f"{'='*60}")
    print("\nController Mapping:")
    print("  Left Stick X:        Base rotation")
    print("  Left Stick Y:        Shoulder joint")
    print("  Right Stick X:       Elbow joint")
    print("  Right Stick Y:       Wrist joint 1")
    print("  L1/R1:              Wrist joints 2/3")
    print("  L2/R2:              Fine control mode (slower movement)")
    print("  Triangle (Y):       Reset to zero")
    print("  Circle (B):         Toggle trajectory")
    print("  Square (X):         Clear trajectory")
    print("  Cross (A):          Exit")
    print("\nTip: Move sticks slowly for precise control")
    print("     Hold L2 or R2 for fine control mode")
    print("="*60 + "\n")
    
    # Initialize robot simulation
    sim = SerialRobotSim()
    sim.setup_axes_3d(title="Robot Control - DualSense Controller")
    sim.ee_traj = []
    
    joint_angles = [0.0] * sim.dof
    joint_names = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Wrist3", "Gripper"]
    
    # Control parameters
    control_rate = 0.02  # 50 Hz control loop
    max_joint_speed = 2.0  # radians per second
    deadzone = 0.1  # Joystick deadzone
    
    # State tracking
    trajectory_recording = True
    last_update_time = time.perf_counter()
    
    print("Controller initialized! Use the controller to move the robot.")
    print("Press Cross (A) button to exit.\n")
    
    # Initial draw
    try:
        sim.draw_pose(joint_angles)
        plt.ioff()
        plt.show(block=False)
    except Exception as e:
        print(f"ERROR: Failed to initialize visualization: {e}")
        pygame.quit()
        return
    
    running = True
    try:
        while running and plt.fignum_exists(sim.fig.number):
            current_time = time.perf_counter()
            dt = current_time - last_update_time
            
            # Process pygame events (important for controller input)
            try:
                pygame.event.pump()  # Process event queue
            except Exception as e:
                print(f"Warning: Event pump error: {e}")
                break
            
            try:
                events = pygame.event.get()
            except Exception as e:
                print(f"Warning: Failed to get events: {e}")
                break
            
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Button mappings (may vary by controller)
                    # DualSense: 0=Cross, 1=Circle, 2=Square, 3=Triangle
                    # Xbox: 0=A, 1=B, 2=X, 3=Y
                    if event.button == 0:  # Cross/A - Exit
                        print("Exiting...")
                        running = False
                    elif event.button == 1:  # Circle/B - Toggle trajectory
                        sim.show_trajectory = not sim.show_trajectory
                        print(f"Trajectory: {'ON' if sim.show_trajectory else 'OFF'}")
                    elif event.button == 2:  # Square/X - Clear trajectory
                        sim.ee_traj = []
                        print("Trajectory cleared")
                    elif event.button == 3:  # Triangle/Y - Reset
                        joint_angles = [0.0] * sim.dof
                        sim.ee_traj = []
                        print("Reset to zero position")
            
            # Read joystick axes
            # Different controllers have different axis layouts
            # DualSense/Xbox: 0=LeftX, 1=LeftY, 2=RightX, 3=RightY, 4=L2, 5=R2
            try:
                num_axes = joystick.get_numaxes()
            except Exception as e:
                print(f"Warning: Failed to get num axes: {e}")
                break
            
            # Left stick (axes 0, 1)
            try:
                left_x = joystick.get_axis(0) if num_axes > 0 else 0.0  # Base rotation
                left_y = -joystick.get_axis(1) if num_axes > 1 else 0.0  # Shoulder (inverted)
            except Exception as e:
                print(f"Warning: Failed to read left stick: {e}")
                left_x = 0.0
                left_y = 0.0
            
            # Right stick - try different axis indices depending on controller type
            # For DualSense/PlayStation controllers, right stick might be at 2,3 or 3,4
            # For Xbox controllers, right stick is typically at 2,3
            try:
                if num_axes >= 4:
                    # Try axes 2,3 first (most common)
                    right_x = joystick.get_axis(2)  # Elbow
                    right_y = -joystick.get_axis(3)  # Wrist1 (inverted)
                elif num_axes >= 3:
                    right_x = joystick.get_axis(2)
                    right_y = 0.0
                else:
                    right_x = 0.0
                    right_y = 0.0
            except Exception as e:
                print(f"Warning: Failed to read right stick: {e}")
                right_x = 0.0
                right_y = 0.0
            
            # Apply deadzone
            def apply_deadzone(value, deadzone_val):
                if abs(value) < deadzone_val:
                    return 0.0
                return value
            
            left_x = apply_deadzone(left_x, deadzone)
            left_y = apply_deadzone(left_y, deadzone)
            right_x = apply_deadzone(right_x, deadzone)
            right_y = apply_deadzone(right_y, deadzone)
            
            # Read triggers (L2/R2) for fine control
            # Triggers are usually axes 4 and 5, or buttons
            fine_control = False
            try:
                if num_axes >= 6:
                    l2 = (joystick.get_axis(4) + 1.0) / 2.0  # Map from [-1,1] to [0,1]
                    r2 = (joystick.get_axis(5) + 1.0) / 2.0
                    fine_control = (l2 > 0.1 or r2 > 0.1)
                elif num_axes >= 5:
                    l2 = (joystick.get_axis(4) + 1.0) / 2.0
                    fine_control = (l2 > 0.1)
            except Exception as e:
                # Triggers might not be available, that's okay
                pass
            
            # Read L1/R1 buttons (usually buttons 4 and 5)
            try:
                num_buttons = joystick.get_numbuttons()
                l1_pressed = num_buttons > 4 and joystick.get_button(4)
                r1_pressed = num_buttons > 5 and joystick.get_button(5)
            except Exception as e:
                l1_pressed = False
                r1_pressed = False
            
            # Calculate joint velocities
            speed_multiplier = 0.3 if fine_control else 1.0
            
            # Update joint angles based on control input
            if sim.dof > 0:
                joint_angles[0] += left_x * max_joint_speed * dt * speed_multiplier  # Base
            if sim.dof > 1:
                joint_angles[1] += left_y * max_joint_speed * dt * speed_multiplier  # Shoulder
            if sim.dof > 2:
                joint_angles[2] += right_x * max_joint_speed * dt * speed_multiplier  # Elbow
            if sim.dof > 3:
                joint_angles[3] += right_y * max_joint_speed * dt * speed_multiplier  # Wrist1
            
            # L1/R1 for wrist joints 2 and 3
            if l1_pressed and sim.dof > 4:
                joint_angles[4] -= max_joint_speed * dt * speed_multiplier  # Wrist2
            if r1_pressed and sim.dof > 5:
                joint_angles[5] += max_joint_speed * dt * speed_multiplier  # Wrist3
            
            # Clamp joint angles to reasonable limits (optional)
            for i in range(sim.dof):
                joint_angles[i] = np.clip(joint_angles[i], -np.pi, np.pi)
            
            # Update visualization
            if dt >= control_rate:
                sim.draw_pose(joint_angles)
                plt.draw()
                plt.pause(0.001)
                last_update_time = current_time
            
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit()
        plt.close(sim.fig)
        print("Controller disconnected. Goodbye!")


def main():

    # These are all "offline" demos without a notion of a control loop, just to
    # show how you can interact with the simulator
    # simple_static_pose_demo()
    # simple_joint_trajectory_demo()
    # simple_ik_trajectory_demo()

    # This is an "online" demo, where the robot is commanded "live" at a certain
    # control loop frequency in the same trajectory as
    # simple_joint_trajectory_demo. You might want to look at:
    #   * q_ref_at: input; this is the input command at time t
    #   * t_log: output; this is a telemetry log of all the timestamps
    #   * q_cmd_log: output; this is a telemetry log of all the commanded joint
    #     positions at time t
    #   * q_act_log: output; this is a telemetry log of all the actual joint
    #     positions at time t
    # simple_control_loop_demo()
    
    # Interactive input demo - uncomment the one you prefer:
    # interactive_command_line_demo()  # Command-line input mode
    # interactive_input_demo()  # Keyboard input mode (requires focus on plot window)
    interactive_controller_demo()  # Controller input mode (DualSense/Xbox/etc)

if __name__ == "__main__":
    main()
