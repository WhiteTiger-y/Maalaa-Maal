import cv2
import streamlit as st
import time
import torch
from gtts import gTTS
import os
from datetime import datetime
import numpy as np

# Streamlit app title and description
st.title("🔍 Real-Time Object Detection with Distance Measurement & Audio Alerts")
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

# Connect to video stream from a specified URL
video_url = "http://192.168.133.38:8081/"  # Replace with actual IP and port
cap = cv2.VideoCapture(video_url)

# Load YOLOv7 model (ensure 'yolov7.pt' path is correct)
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='yolov7.pt', source='github')

# Function to calculate distance based on bounding box width
def calculate_distance(width_in_frame, actual_width=0.5, focal_length=615):
    """Estimate distance to object in meters."""
    if width_in_frame > 0:
        return (actual_width * focal_length) / width_in_frame
    return None

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
        if time.time() - last_capture_time >= 10:
            last_capture_time = time.time()

            # Perform object detection on the current frame
            results = model(frame)
            detections = results.pred[0].cpu().numpy()  # Get predictions
            names = results.names  # Access class names from model

            detection_entries = []  # To store detection texts for audio and display

            # Define the frame's width and boundaries for segmentation
            frame_width = frame.shape[1]
            left_boundary = frame_width // 3
            right_boundary = left_boundary * 2

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.5:  # Only consider high-confidence detections
                    obj_name = names[int(cls)]
                    width_in_frame = x2 - x1

                    # Calculate distance to the detected object
                    distance = calculate_distance(width_in_frame)
                    detected_text = f"{obj_name} at {distance:.2f}m" if distance else f"{obj_name} (distance unknown)"

                    # Determine object position (left, middle, or right)
                    center_x = (x1 + x2) / 2
                    if center_x < left_boundary:
                        position = "on your left"
                    elif center_x < right_boundary:
                        position = "in front of you"
                    else:
                        position = "on your right"
                    
                    # Add position to detection text
                    detected_text += f" {position}"
                    detection_entries.append(detected_text)

                    # Draw bounding box and distance on frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{obj_name}: {distance:.2f}m {position}" if distance else f"{obj_name}: Calculating... {position}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the updated frame in Streamlit
            video_placeholder.image(frame, channels="BGR")

            if detection_entries:
                # Add timestamp to detections
                detection_time = datetime.now().strftime("%H:%M:%S")
                detection_text = f"{detection_time} - Detected: {', '.join(detection_entries)}"

                # Update detection history in session state and limit to last 5 entries
                st.session_state.detection_texts.append(detection_text)
                st.session_state.detection_texts = st.session_state.detection_texts[-5:]

                detection_text= f"Detected: {', '.join(detection_entries)}"

                # Display detection history
                with detection_history_container:
                    detection_history_container.empty()
                    st.markdown("### Detection History")
                    for text in reversed(st.session_state.detection_texts):
                        st.write(text)

                # Generate audio message if enabled
                if audio_alert:
                    tts = gTTS(detection_text)
                    tts.save("detection.mp3")
                    os.system("start detection.mp3")  # Windows; adjust as needed for Mac/Linux

# Release resources
cap.release()
