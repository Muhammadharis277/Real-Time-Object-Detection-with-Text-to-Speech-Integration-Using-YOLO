import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import threading
import pyttsx3

# Streamlit page configuration
st.title("Real-Time Object Detection with Text-to-Speech")

# Load YOLO model (your custom model or a pre-trained one for testing)
model = YOLO('yolo11n.pt')  # Change to your custom model if necessary

# Function to handle TTS in a thread
def speak_text(text):
    # Initialize TTS engine inside the thread to avoid resource conflicts
    tts_engine = pyttsx3.init()
    tts_engine.say(text)
    tts_engine.runAndWait()

# Buttons for controlling the webcam
start_detection = st.button("Start Detection 🚦", key="start", use_container_width=True)
stop_detection = st.button("Stop Detection 🛑", key="stop", use_container_width=True)

# Streamlit placeholders for displaying frames, confidence scores, and classes
frame_placeholder = st.empty()
info_placeholder = st.empty()

# Global flag to control detection
is_running = False
last_said_class = None  # To prevent repeated announcements

# Start detection if "Start Detection" is clicked
if start_detection:
    is_running = True

# Loop for real-time detection
if is_running:
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        exit()

    # Set the frame resolution
    frame_width = 640
    frame_height = 480
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    while is_running and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to grab frame, exiting.")
            break

        # Run object detection on the frame
        results = model(frame)  # Perform object detection

        detected_info = []  # To store detected objects' info
        max_confidence = 0
        top_class = None

        # Check if any detections were made
        if results[0].boxes is None:
            detected_info.append("No detections found!")
        else:
            # Extract detected objects
            for result in results[0].boxes:
                conf = float(result.conf[0])  # Confidence score
                cls = int(result.cls[0])  # Class ID
                class_name = model.names[cls] if model.names else f"Class {cls}"  # Class name
                if conf > 0.5:  # Filter low-confidence detections
                    xywh = result.xywh[0]  # [center_x, center_y, width, height]
                    detected_info.append(f"Detected: {class_name}, Confidence: {conf:.2f}")
                    if conf > max_confidence:
                        max_confidence = conf
                        top_class = class_name

        # If a high-confidence class is detected and it's not the same as the last spoken class, announce it
        if top_class and top_class != last_said_class:
            last_said_class = top_class
            threading.Thread(target=speak_text, args=(f"{top_class}",)).start()

        # Draw the detections on the frame
        frame_with_detections = results[0].plot()  # Draw the boxes on the frame

        # Convert the frame to RGB (Streamlit requires RGB format)
        frame_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)

        # Update the placeholders
        frame_placeholder.image(frame_rgb, channels="RGB")
        info_placeholder.text("\n".join(detected_info))  # Update detections dynamically

        # Stop detection if "Stop Detection" is clicked
        if stop_detection:
            is_running = False
            break

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
    st.write("Webcam feed stopped.")

else:
    st.write("Click the 'Start Detection' button to begin.")
