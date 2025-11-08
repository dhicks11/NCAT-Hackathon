#!/usr/bin/env python3
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# -------------------------------
# Telemetry Recorder
# -------------------------------
class TelemetryRecorder:
    def __init__(self):
        self.data = []

    def record(self, joint_positions, ee_position, ee_orientation, current, torque, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.data.append({
            "timestamp": timestamp,
            "joint_positions": joint_positions,
            "ee_position": ee_position,
            "ee_orientation": ee_orientation,
            "current": current,
            "torque": torque
        })

    def save_csv(self, filename="telemetry.csv"):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

    def load_csv(self, filename):
        df = pd.read_csv(filename)
        self.data = df.to_dict('records')

# -------------------------------
# Simulator (Serial 7-DOF Robot)
# -------------------------------
def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ], dtype=float)

class SerialRobotSim:
    def __init__(self):
        self.dof = 7
        self.a = [7,6,4,5,3,2,1.5]
        self.alpha = [np.pi/2,0,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,0]
        self.d = [0]*self.dof
        self.fig = None
        self.ax = None
        self.joint_lines = None
        self.ee_traj = []

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        positions = [(0,0,0)]
        for i in range(self.dof):
            Ti = dh_transform(self.a[i], self.alpha[i], self.d[i], joint_angles[i])
            T = T @ Ti
            positions.append(tuple(T[:3,3]))
        return positions

    def demo_inverse_kinematics(self, x, y, z):
        q = [0.0]*self.dof
        q[0] = np.arctan2(y, x)
        r = np.hypot(x, y)
        zt = z
        L0, L1 = self.a[0], self.a[1]
        R = np.clip(np.hypot(r, zt), 1e-9, L0+L1-1e-9)
        cos_elbow = np.clip((R**2-L0**2-L1**2)/(2*L0*L1), -1,1)
        elbow = np.arccos(cos_elbow)
        gamma = np.arctan2(zt,r)
        phi = np.arctan2(L1*np.sin(elbow), L0+L1*np.cos(elbow))
        q[1] = gamma - phi
        q[2] = elbow
        # wrist neutral
        return q

    def setup_axes_3d(self):
        reach = sum(self.a)
        self.fig = plt.figure(figsize=(8,6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-reach, reach)
        self.ax.set_ylim(-reach, reach)
        self.ax.set_zlim(0, reach)
        self.ax.set_xlabel('X'); self.ax.set_ylabel('Y'); self.ax.set_zlabel('Z')
        self.joint_lines = [self.ax.plot([],[],[],'o-', linewidth=2, markersize=4)[0] for _ in range(self.dof)]
        self.ee_traj = []

    def draw_pose(self, joint_angles):
        positions = self.forward_kinematics(joint_angles)
        for j, line in enumerate(self.joint_lines):
            a = positions[j]; b = positions[j+1]
            xs, ys, zs = zip(a,b)
            line.set_data(xs,ys)
            line.set_3d_properties(zs)
        # trajectory
        ee = positions[-1]
        self.ee_traj.append(ee)
        tx, ty, tz = (np.array(t) for t in zip(*self.ee_traj))
        if hasattr(self,'traj_line')==False:
            self.traj_line, = self.ax.plot(tx,ty,tz,'r--', linewidth=1.2)
        else:
            self.traj_line.set_data(tx,ty)
            self.traj_line.set_3d_properties(tz)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(layout="wide")
st.title("🤖 RoboMaster TeleOp Panel")

# Instantiate
sim = SerialRobotSim()
sim.setup_axes_3d()
recorder = TelemetryRecorder()

# Sidebar
st.sidebar.header("Task-space control")
x = st.sidebar.slider("X", -10, 10, 0.0)
y = st.sidebar.slider("Y", -10, 10, 0.0)
z = st.sidebar.slider("Z", 0, 10, 5.0)

st.sidebar.header("Joint-space control")
joint_angles = []
for i in range(sim.dof):
    joint_angles.append(st.sidebar.slider(f"Joint {i+1}", -np.pi, np.pi, 0.0))

# Task-space overrides joint-space via IK
ik_joint_angles = sim.demo_inverse_kinematics(x, y, z)

# Draw robot pose
sim.draw_pose(ik_joint_angles)

# Record telemetry
recorder.record(
    joint_positions=ik_joint_angles,
    ee_position=sim.forward_kinematics(ik_joint_angles)[-1],
    ee_orientation=(0,0,0),
    current=[0]*sim.dof,
    torque=[0]*sim.dof
)

# Show robot figure
st.pyplot(sim.fig)

# Save telemetry
if st.button("Save Telemetry"):
    recorder.save_csv()
    st.success("Telemetry saved as telemetry.csv!")

# Replay telemetry
st.header("Replay Telemetry")
uploaded_file = st.file_uploader("Upload CSV to replay", type="csv")
if uploaded_file:
    recorder.load_csv(uploaded_file)
    replay_fig = plt.figure(figsize=(8,6))
    replay_ax = replay_fig.add_subplot(111, projection='3d')
    replay_ax.set_xlim(-sum(sim.a), sum(sim.a))
    replay_ax.set_ylim(-sum(sim.a), sum(sim.a))
    replay_ax.set_zlim(0, sum(sim.a))
    replay_ax.set_xlabel('X'); replay_ax.set_ylabel('Y'); replay_ax.set_zlabel('Z')
    for entry in recorder.data:
        pos = sim.forward_kinematics(entry["joint_positions"])
        xs, ys, zs = zip(*pos)
        replay_ax.plot(xs, ys, zs, 'o-', alpha=0.3)
    st.pyplot(replay_fig)
