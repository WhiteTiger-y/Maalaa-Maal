import streamlit as st
import torch
import cv2
import numpy as np
import time

# Load YOLOv7 model (ensure 'yolov7.pt' path is correct)
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model='yolov7.pt', source='github')

# Streamlit app setup
st.title("YOLOv7 Distance Measurement")
st.text("Detect objects and measure their distances in real-time.")

# Function to calculate distance
def calculate_distance(width_in_frame, actual_width, focal_length):
    if width_in_frame > 0:
        return (actual_width * focal_length) / width_in_frame
    return None

# Function to process and display detections with distances
def process_frame(frame, focal_length, actual_width):
    results = model(frame)  # Perform detection
    detections = results.pred[0].cpu().numpy()  # Get predictions
    names = results.names  # Access the class names from the model
    annotated_frame = frame.copy()

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.5:  # Threshold to ignore low-confidence detections
            width_in_frame = x2 - x1
            distance = calculate_distance(width_in_frame, actual_width, focal_length)
            
            # Get the class label from the model's result
            class_label = names[int(cls)] if int(cls) < len(names) else 'Unknown'
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Annotate distance and class
            label = f"{class_label}: {distance:.2f}m" if distance else class_label
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
    return annotated_frame

# Set parameters for distance calculation
focal_length = 615  # Example focal length; adjust based on your camera
actual_width = 0.5  # Approximate width of object in meters

# Load video stream URL
video_url = "http://192.168.133.38:8081/"

# Streamlit video capture display
stframe = st.empty()
video_capture = cv2.VideoCapture(video_url)

# Process each frame
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to fetch video frame.")
        break
    
    # Process frame for detection and distance annotation
    processed_frame = process_frame(frame, focal_length, actual_width)
    
    # Display the frame in Streamlit
    stframe.image(processed_frame, channels="BGR")
    
    # Small delay to maintain stream rate
    time.sleep(0.1)

# Release video capture on app exit
video_capture.release()
