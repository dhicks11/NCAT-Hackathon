import streamlit as st
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from performance_metrics import PerformanceMetrics

# Get the directory where this script is located (needed early for paths)
SCRIPT_DIR = Path(__file__).parent

# -----------------------
# Telemetry Recorder
# -----------------------
class TelemetryRecorder:
    def __init__(self):
        self.data = []

    def record(self, joint_positions, ee_position, ee_orientation=(0,0,0), current=None, torque=None, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        if current is None:
            current = [0.0]*len(joint_positions)
        if torque is None:
            torque = [0.0]*len(joint_positions)
        entry = {
            "timestamp": timestamp,
            "joint_positions": joint_positions,
            "ee_position": ee_position,
            "ee_orientation": ee_orientation,
            "current": current,
            "torque": torque
        }
        self.data.append(entry)

    def save_csv(self, filename=None):
        if not self.data:
            return None
        if filename is None:
            # Save in the script directory
            filename = str(SCRIPT_DIR / "telemetry.csv")
        rows = []
        for e in self.data:
            rows.append({
                "timestamp": e["timestamp"],
                "joint_positions": json.dumps(e["joint_positions"]),
                "ee_position": json.dumps(e["ee_position"]),
                "ee_orientation": json.dumps(e["ee_orientation"]),
                "current": json.dumps(e["current"]),
                "torque": json.dumps(e["torque"])
            })
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        return filename

    def load_csv(self, filename):
        df = pd.read_csv(filename)
        loaded = []
        for _, r in df.iterrows():
            loaded.append({
                "timestamp": float(r["timestamp"]),
                "joint_positions": json.loads(r["joint_positions"]),
                "ee_position": json.loads(r["ee_position"]),
                "ee_orientation": json.loads(r["ee_orientation"]),
                "current": json.loads(r["current"]),
                "torque": json.loads(r["torque"])
            })
        self.data = loaded

# -----------------------
# Mock ROS (simulation-only)
# -----------------------
class MockROS:
    def __init__(self):
        self.topics = {}
        self.log = []

    def publish(self, topic, msg):
        t = time.time()
        self.log.append((t, topic, msg))
        if topic in self.topics:
            for cb in self.topics[topic]:
                try:
                    cb(msg)
                except Exception:
                    pass

    def subscribe(self, topic, callback):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(callback)

    def get_log(self, n=50):
        return self.log[-n:]

# -----------------------
# Forward kinematics (simple 4-joint robot)
# -----------------------
def forward_kinematics_deg(t1, t2, t3, t4, L1=4.0, L2=3.0, L3=2.0):
    t1, t2, t3, t4 = np.radians([t1, t2, t3, t4])
    x0, y0, z0 = 0,0,0
    x1 = L1*np.cos(t1)
    y1 = L1*np.sin(t1)
    z1 = 0
    x2 = x1 + L2*np.cos(t1)*np.cos(t2)
    y2 = y1 + L2*np.sin(t1)*np.cos(t2)
    z2 = L2*np.sin(t2)
    x3 = x2 + L3*np.cos(t1)*np.cos(t2+t3)
    y3 = y2 + L3*np.sin(t1)*np.cos(t2+t3)
    z3 = z2 + L3*np.sin(t2+t3)
    x4 = x3 + 0.5*np.cos(t1)*np.cos(t2+t3+t4)
    y4 = y3 + 0.5*np.sin(t1)*np.cos(t2+t3+t4)
    z4 = z3 + 0.5*np.sin(t2+t3+t4)
    return [[x0,y0,z0],[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]]

# -----------------------
# Paths
# -----------------------
TEMPLATES_DIR = SCRIPT_DIR / "templates"
INDEX_HTML_PATH = TEMPLATES_DIR / "index.html"
if not INDEX_HTML_PATH.exists():
    raise FileNotFoundError(f"Missing {INDEX_HTML_PATH}. Put index.html inside templates/")

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="SurgiControl", layout="wide", page_icon="🦾")
st.title("🩺 SurgiControl — Sim + Mock ROS")

# Session state
if "recorder" not in st.session_state:
    st.session_state.recorder = TelemetryRecorder()
if "ros" not in st.session_state:
    st.session_state.ros = MockROS()
if "last_auto_save" not in st.session_state:
    st.session_state.last_auto_save = time.time()
if "auto_recording" not in st.session_state:
    st.session_state.auto_recording = True
if "metrics" not in st.session_state:
    st.session_state.metrics = PerformanceMetrics()

recorder: TelemetryRecorder = st.session_state.recorder
mockros: MockROS = st.session_state.ros

# Layout
col_left, col_right = st.columns([1,2])

with col_left:
    st.header("Controls")
    
    # Controller instructions
    with st.expander("🎮 Controller Control Instructions", expanded=False):
        st.markdown("""
        **To control the robot with your DualSense/Xbox controller:**
        
        1. **Run the standalone controller demo:**
           ```bash
           python run_controller_demo.py
           ```
        
        2. **Controller Mapping:**
           - **Left Stick X** → Base rotation (left/right)
           - **Left Stick Y** → Shoulder joint (up/down)
           - **Right Stick X** → Elbow joint (left/right)
           - **Right Stick Y** → Wrist joint (up/down)
           - **L1/R1** → Wrist joints 2/3
           - **L2/R2** → Fine control mode (slower)
           - **Triangle (Y)** → Reset to zero
           - **Circle (B)** → Toggle trajectory
           - **Square (X)** → Clear trajectory
           - **Cross (A)** → Exit
        
        3. **Tips:**
           - Move sticks slowly for precise control
           - Hold L2 or R2 for fine control mode
           - The controller demo opens a 3D visualization window
        """)
    
    # Normal slider controls
    j1 = st.slider("Base (°)", -180.0, 180.0, 0.0, step=0.5)
    j2 = st.slider("Shoulder (°)", -90.0, 90.0, 0.0, step=0.5)
    j3 = st.slider("Elbow (°)", -90.0, 90.0, 0.0, step=0.5)
    j4 = st.slider("Wrist (°)", -90.0, 90.0, 0.0, step=0.5)
    
    # Periodic telemetry recording (every 30 seconds)
    current_time = time.time()
    if st.session_state.auto_recording and (current_time - st.session_state.last_auto_save) >= 30:
        ee_positions = forward_kinematics_deg(j1, j2, j3, j4)
        ee = ee_positions[-1]
        # Generate mock torque and trajectory data
        torque_data = [abs(j) * 0.1 for j in [j1, j2, j3, j4]]  # Mock torque based on joint angles
        trajectory_points = ee_positions  # Full trajectory from base to end-effector
        
        recorder.record([j1, j2, j3, j4], ee, torque=torque_data)
        # Auto-save to dataset
        dataset_file = recorder.save_csv(str(SCRIPT_DIR / f"telemetry_dataset_{int(current_time)}.csv"))
        st.session_state.last_auto_save = current_time
        st.info(f"Auto-recorded: joints={[j1,j2,j3,j4]}, ee={ee}, torque={torque_data}, trajectory={len(trajectory_points)} points")

    # Telemetry
    st.session_state.auto_recording = st.checkbox("Auto-record every 30s", value=st.session_state.auto_recording)
    
    if st.button("Record current frame"):
        ee_positions = forward_kinematics_deg(j1,j2,j3,j4)
        ee = ee_positions[-1]
        torque_data = [abs(j) * 0.1 for j in [j1, j2, j3, j4]]
        recorder.record([j1,j2,j3,j4], ee, torque=torque_data)
        st.success(f"Recorded: joints={[j1,j2,j3,j4]}, torque={torque_data}")

    if st.button("Save telemetry to CSV"):
        fname = recorder.save_csv()
        if fname:
            st.success(f"Saved telemetry to {fname}")
        else:
            st.warning("No telemetry to save yet")

    uploaded = st.file_uploader("Upload CSV for replay", type=["csv"])
    if uploaded:
        # Use a temporary file in the script directory or system temp
        import tempfile
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        recorder.load_csv(tmp_path)
        # Clean up temp file after loading
        try:
            Path(tmp_path).unlink()
        except:
            pass
        st.success("Loaded telemetry")

    st.markdown("---")
    st.subheader("Mock ROS")
    if st.button("Publish current joint command to /cmd_joints"):
        msg = {"joints":[j1,j2,j3,j4],"stamp":time.time()}
        mockros.publish("/cmd_joints", msg)
        st.success("Published")

    st.write("ROS log (latest 10):")
    for t,topic,msg in mockros.get_log(10):
        st.write(f"{time.strftime('%H:%M:%S', time.localtime(t))} {topic} {msg}")

with col_right:
    st.header("3D Simulation")
    ee_positions = forward_kinematics_deg(j1,j2,j3,j4)
    
    # Prepare trajectory data - ensure it's a list of [x, y, z] lists
    trajectory_data = []
    for e in recorder.data:
        if isinstance(e.get("ee_position"), list) and len(e["ee_position"]) >= 3:
            trajectory_data.append(e["ee_position"])
    
    init_state = {
        "joints":[float(j1), float(j2), float(j3), float(j4)],
        "ee": ee_positions[-1] if ee_positions else [0, 0, 0],
        "trajectory": trajectory_data
    }

    html_code = INDEX_HTML_PATH.read_text(encoding="utf-8")
    # Replace the placeholder with actual data
    init_data_js = f"const init_state = {json.dumps(init_state)};"
    injected = html_code.replace("/*__INIT_DATA__*/", init_data_js)
    
    # Render the 3D visualization (Streamlit will automatically update when sliders change)
    st.components.v1.html(injected, height=720, scrolling=False)

st.markdown("---")
st.caption("SurgiControl — simulated robot + mock ROS demo")
st.caption("© 2024 ByteBoyz. All rights reserved.")
