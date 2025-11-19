# ==============================================================================
# FRONTEND - STREAMLIT USER INTERFACE (v4 - Final)
# ==============================================================================

import streamlit as st
import cv2
import time
import json
from datetime import datetime
from backend import AIObjectDetector, AICameraController, Auto3DMeasurement

st.set_page_config(page_title="AI 3D Object Measurement", layout="wide")
if 'camera_connected' not in st.session_state: st.session_state.camera_connected = False
if 'camera_source' not in st.session_state: st.session_state.camera_source = None

@st.cache_resource
def get_ai_detector(): return AIObjectDetector()

try:
    detector = get_ai_detector()
    measurement_engine = Auto3DMeasurement(detector)
    st.title("🤖 AI 3D Object Measurement System (YOLOv8)")
    st.markdown("A web-based interface to perform 3D measurements using a locally stored YOLOv8 model.")
    with st.expander("📖 How to Use This Application", expanded=True):
        st.markdown("""
            1.  **Connect to Camera**: Use the sidebar to select and connect to your Webcam or IP Camera.
            2.  **Start Measurement**: Click the "Start Measurement" button.
            3.  **Capture & Move**: The app captures the first image, then waits for you to move the camera horizontally to the right.
            4.  **Get Results**: After the second capture, the AI calculates and displays the 3D measurements.
        """)
    st.sidebar.title("⚙️ Controls & Settings")
    st.sidebar.subheader("1. Connect to Camera")
    cam_type = st.sidebar.radio("Choose Camera Source", ("Default Webcam", "IP Camera"))
    camera_input = "0" if cam_type == "Default Webcam" else st.sidebar.text_input("Enter IP Camera Address", "http://192.168.1.10:8080/video")
    if st.sidebar.button("Connect"):
        if not camera_input: st.sidebar.error("Please provide a valid camera input.")
        else:
            with st.spinner(f"Connecting to {camera_input}..."):
                try:
                    frame = AICameraController(camera_input).capture_single_image()
                    if frame is not None: st.session_state.camera_connected = True; st.session_state.camera_source = camera_input; st.sidebar.success(f"Connected to: {camera_input}")
                    else: st.session_state.camera_connected = False; st.sidebar.error("Connection failed.")
                except Exception as e: st.session_state.camera_connected = False; st.sidebar.error(f"Failed to connect: {e}")
    if st.session_state.camera_connected: st.sidebar.success(f"✅ Connected to: {st.session_state.camera_source}")
    else: st.sidebar.warning("⚠️ Not connected to any camera.")
    st.sidebar.subheader("2. Start Measurement")
    capture_delay = st.sidebar.slider("Capture Delay (seconds)", 3, 10, 6)
    if st.sidebar.button("🚀 Start Measurement", type="primary", use_container_width=True, disabled=not st.session_state.camera_connected):
        col1, col2 = st.columns(2); left_image_placeholder, right_image_placeholder, status_placeholder = col1.empty(), col2.empty(), st.empty()
        try:
            camera_controller = AICameraController(st.session_state.camera_source)
            status_placeholder.info("Capturing LEFT image..."); left_image = camera_controller.capture_single_image(); left_image_placeholder.image(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB), caption="Left Image", use_column_width=True); status_placeholder.success("✅ Left image captured!")
            progress_bar = st.progress(0); status_placeholder.info(f"Move camera RIGHT. Waiting {capture_delay}s...")
            for i in range(capture_delay): time.sleep(1); progress_bar.progress((i + 1) / capture_delay)
            progress_bar.empty()
            status_placeholder.info("Capturing RIGHT image..."); right_image = camera_controller.capture_single_image(); right_image_placeholder.image(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB), caption="Right Image", use_column_width=True); status_placeholder.success("✅ Right image captured!")
            st.subheader("Processing and Measurement")
            with st.spinner("AI (YOLOv8) is analyzing images..."): measurements, result_image = measurement_engine.run_full_measurement_process(left_image, right_image)
            st.subheader("Final Results")
            if measurements:
                st.success("✅ Measurement complete!"); st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Annotated Measurement Results", use_column_width=True); st.markdown("#### Detailed Measurements Table:"); st.dataframe(measurements)
                results_json = json.dumps(measurements, indent=2); st.download_button(label="💾 Download Results as JSON", data=results_json, file_name=f"ai_measurements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
            else: st.error("❌ Measurement failed. No objects could be reliably matched.")
        except Exception as e: st.error(f"An error occurred: {e}")
except RuntimeError as e:
    st.error(f"🔴 APPLICATION FAILED TO START: {e}")
    st.error("Please ensure the 'yolov8m.pt' file is in the project directory and that you have run 'pip install -r requirements.txt' in a clean virtual environment.")
