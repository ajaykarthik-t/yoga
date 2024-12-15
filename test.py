import streamlit as st
import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load the trained model
model = joblib.load('yoga_pose_classifier.pkl')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Title
st.title("Real-Time Yoga Pose Detection with Live Video Feed")

# Placeholder for video
video_placeholder = st.empty()

# Start/Stop buttons
if "start_detection" not in st.session_state:
    st.session_state.start_detection = False

if not st.session_state.start_detection:
    if st.button("Start Detection"):
        st.session_state.start_detection = True
else:
    if st.button("Stop Detection"):
        st.session_state.start_detection = False

# Real-time pose detection
def yoga_pose_detection():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while st.session_state.start_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access webcam. Please check your device.")
            break

        # Convert frame to RGB (MediaPipe requires RGB input)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Overlay detections
        pose_label = "No Pose Detected"
        if results.pose_landmarks:
            # Extract landmarks and predict the pose
            landmarks = [landmark.x for landmark in results.pose_landmarks.landmark] + \
                         [landmark.y for landmark in results.pose_landmarks.landmark]
            landmarks = np.array(landmarks).reshape(1, -1)
            pose_label = model.predict(landmarks)[0]

            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display pose label on the video frame
        cv2.putText(frame, f"Pose: {pose_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame for Streamlit and display
        video_placeholder.image(frame, channels="BGR")

    # Release webcam when detection is stopped
    cap.release()

# Run detection when "Start Detection" is pressed
if st.session_state.start_detection:
    yoga_pose_detection()
