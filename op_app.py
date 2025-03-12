import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import os
from moviepy import  ImageSequenceClip
from ultralytics import YOLO
import requests
from PIL import Image
import io

# Load YOLOv8 Model (Pretrained for object detection)
yolo_model = YOLO("yolov8n.pt")

# Load Action Recognition Model
action_model = tf.keras.models.load_model("model.h5")

# Action Classes
action_classes = ['BabyCrawling', 'Dancing', 'Drinking', 'Smoking', 'Hitting',
                  'GunShooting', 'CuttingInKitchen', 'FallingOnFloor', 'Talking', 'WritingOnBoard']

# Streamlit UI
st.title("🎥 Action Recognition Dashboard")

# Sidebar for Real-Time Recognized Action
st.sidebar.header("📌 Live Recognized Action")
last_action = st.sidebar.empty()  # Placeholder for recognized action

# Variable to store last detected action
recognized_action = "Waiting for detection..."


def save_video(frames, output_path, fps=10):
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], fps=fps)
    clip.write_videofile(output_path, codec='libx264')

# Function to Extract Frames from Video
def extract_frames(video_path, frame_step=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        for _ in range(frame_step - 1):
            cap.grab()  # Skip frames
    cap.release()
    return frames

# Function to Process Frames with YOLO + Action Recognition
def process_frames(frames, frame_sequence=10):
    global recognized_action  # Maintain across function calls
    processed_frames = []
    buffer = []

    for frame in frames:
        # Run YOLO Object Detection
        results = yolo_model(frame)
        detections = results[0].boxes.xyxy  # Bounding boxes
        labels = results[0].boxes.cls  # Class labels

        # Resize frame for Action Recognition Model
        frame_resized = cv2.resize(frame, (224, 224)) / 255.0
        buffer.append(frame_resized)

        # If we have enough frames, classify the action
        if len(buffer) == frame_sequence:
            action_pred = action_model.predict(np.expand_dims(buffer, axis=0))
            recognized_action = action_classes[np.argmax(action_pred)]  # Update global action
            buffer = []

        # Display recognized action in sidebar (only updates when an action is detected)
        last_action.write(f"🟢 Recognized Action: **{recognized_action}**")

        # Draw YOLO Bounding Boxes & Action Label
        for box, label in zip(detections, labels):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, recognized_action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        processed_frames.append(frame)

    return processed_frames

# Function for Live Stream Processing
def process_live_stream(stream_url):
    # Function to Capture Images
    def capture_images(image_url, num_images=20):
        images = []
        for _ in range(num_images):
            response = requests.get(image_url, verify=False)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                images.append(image)
            else:
                st.warning("Failed to retrieve image.")
        return images

# Function to Create Video from Images
    def create_video(images, output_path="output.mp4", fps=50):
    # Convert images to numpy arrays and resize
        frame_array = [np.array(img.resize((640, 480))) for img in images]
    
    # Create a video clip from the image sequence
        clip = ImageSequenceClip(frame_array, fps=fps)
        clip.write_videofile(output_path, codec='libx264')
        return output_path
    images = capture_images(stream_url, 50)
    if images:
        video_path = create_video(images)
               # Process the created video
        frames = extract_frames(video_path)
        processed_frames = process_frames(frames)
        output_video_path = "processed_video.mp4"
        save_video(processed_frames, output_video_path)
        st.video(output_video_path)
        st.download_button("⬇️ Download Processed Video", open(output_video_path, "rb").read(), file_name="processed_video.mp4")

# Choose Input Method
option = st.radio("Choose Input Method:", ("Upload a Video", "Live Stream URL"))

# Option 1: Upload Video
if option == "Upload a Video":
    uploaded_file = st.file_uploader("📂 Upload an MP4 Video", type=["mp4"])
    
    if uploaded_file:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(temp_video_path)  # Display uploaded video

        if st.button("Process Video"):
            frames = extract_frames(temp_video_path)
            processed_frames = process_frames(frames)

            output_video_path = "processed_video.mp4"
            save_video(processed_frames, output_video_path)

            st.video(output_video_path)
            st.download_button("⬇️ Download Processed Video", open(output_video_path, "rb").read(), file_name="processed_video.mp4")

# Option 2: Live Stream URL
elif option == "Live Stream URL":
    stream_url = st.text_input("📡 Enter Live Stream URL (RTSP, HTTP, or YouTube)")
    
    if st.button("Start Stream"):
        process_live_stream(stream_url)
