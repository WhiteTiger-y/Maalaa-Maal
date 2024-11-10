import cv2
import streamlit as st
import time
import torch
from gtts import gTTS
import os
from datetime import datetime
from gradio_client import Client
from gradio_client.data_classes import FileData

# Streamlit app title and description
st.title("🔍 Real-Time Object Detection with Audio Alerts")
st.sidebar.header("Settings")
st.sidebar.write("Configure the options for the app below.")

# Toggle for audio alerts
audio_alert = st.sidebar.checkbox("Enable Audio Alerts", value=True)

# Placeholder for video feed and detection history
video_placeholder = st.empty()
detection_history_container = st.container()  # Container for showing detection history

# Initialize detection history if not already done
if "detection_texts" not in st.session_state:
    st.session_state.detection_texts = []

detection_sents=[]

# Connect to video stream from a specified URL
video_url = "http://192.168.133.38:8081/"  # Replace with actual IP and port
cap = cv2.VideoCapture(video_url)

# Load YOLO model (assumes YOLOv5 in PyTorch)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Ensure YOLO is installed

# def predict_space(frame, model="ViT-GPT2"):
#     frame_path = "temp_frame.jpg"
#     cv2.imwrite(frame_path, frame)
#     client = Client("kanishk128/eye_for_blind")
    
#     # Pass the file path directly as a string to avoid validation issues
#     result = client.predict(
#         frame_path,   # Direct file path as string
#         model,        # Model selection
#         api_name="/predict"
#     )
#     return result

# Check if video source is accessible
if not cap.isOpened():
    st.error("Unable to connect to video source.")
else:
    last_capture_time = time.time()

    # Stream frames from the video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to retrieve frame.")
            break

        # Convert frame to RGB for display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the video feed in Streamlit
        video_placeholder.image(display_frame, channels="RGB")

        # Check if 5 seconds have passed for the next capture
        if time.time() - last_capture_time >= 5:
            last_capture_time = time.time()

            # Perform object detection on the current frame
            results = model(frame)
            detected_objects = results.pandas().xyxy[0]['name'].tolist()

            if detected_objects:
                # Format detected objects and add timestamp
                detection_time = datetime.now().strftime("%H:%M:%S")
                detected_text = f"{detection_time} - Detected: {', '.join(detected_objects)}"
                
                # Append to detection history in session state and limit to last 5 entries
                st.session_state.detection_texts.append(detected_text)
                st.session_state.detection_texts = st.session_state.detection_texts[-5:]
                # detection_sents.append(predict_space(frame))
                print(detection_sents)

                # Display the last 5 detection events in reverse order
                with detection_history_container:
                    # Clear the container before updating
                    detection_history_container.empty()
                    st.markdown("### Detection History")
                    for text in reversed(st.session_state.detection_texts):  # Reverse order for latest-first
                        st.write(text)

                # Generate audio message if enabled
                if audio_alert:
                    tts = gTTS(detected_text)
                    tts.save("detection.mp3")
                    os.system("start detection.mp3")  # Windows; adjust as needed for Mac/Linux

# Release resources
cap.release()

