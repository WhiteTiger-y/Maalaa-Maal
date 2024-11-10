import cv2
import streamlit as st
import time
import torch
from gtts import gTTS
import os
from datetime import datetime

# Streamlit app title and description
st.title("🔍 Real-Time Object Detection with Audio Alerts")
st.sidebar.header("Settings")
st.sidebar.write("Configure the options for the app below.")

# Toggle for audio alerts
audio_alert = st.sidebar.checkbox("Enable Audio Alerts", value=True)

# Placeholder for video feed and detection history
video_placeholder = st.empty()
detection_history_container = st.container()

# Initialize detection history if not already done
if "detection_texts" not in st.session_state:
    st.session_state.detection_texts = []

# Connect to video stream
video_url = "http://192.168.133.38:8081/"  # Replace with actual IP and port
cap = cv2.VideoCapture(video_url)

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='yolov7.pt', source='github')

# Check if video source is accessible
if not cap.isOpened():
    st.error("Unable to connect to video source.")
else:
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to retrieve frame.")
            break

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame, channels="RGB")

        # Only perform object detection every 5 seconds
        if time.time() - last_capture_time >= 5:
            last_capture_time = time.time()

            results = model(frame)
            detections = results.pred[0].cpu().numpy()
            names = results.names
            frame_width = frame.shape[1]
            left_boundary = frame_width // 3
            right_boundary = left_boundary * 2

            detected_texts = []
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.5:
                    obj_name = names[int(cls)]
                    center_x = (x1 + x2) / 2

                    # Determine the position (left, middle, or right)
                    if center_x < left_boundary:
                        position = "on your left"
                    elif center_x < right_boundary:
                        position = "in the middle"
                    else:
                        position = "on your right"

                    detected_text = f"{obj_name} {position}"
                    detected_texts.append(detected_text)
            
            if detected_texts:
                detection_time = datetime.now().strftime("%H:%M:%S")
                detected_summary = f"{detection_time} - Detected: {', '.join(detected_texts)}"
                
                # Update detection history
                st.session_state.detection_texts.append(detected_summary)
                st.session_state.detection_texts = st.session_state.detection_texts[-5:]

                with detection_history_container:
                    detection_history_container.empty()
                    st.markdown("### Detection History")
                    for text in reversed(st.session_state.detection_texts):
                        st.write(text)

                if audio_alert:
                    tts = gTTS(detected_summary)
                    tts.save("detection.mp3")
                    os.system("start detection.mp3")  # Windows; adjust as needed for Mac/Linux

cap.release()
