This project integrates the YOLO (You Only Look Once) algorithm with a Text-to-Speech (TTS) engine to develop a real-time object detection system that provides auditory feedback about detected objects. The system uses a pre-trained YOLOv11 model, eliminating the need for custom datasets and focusing on efficient deployment and application. Designed for live testing using a user-friendly graphical user interface (GUI) built with Streamlit, the project highlights the practical applications of computer vision in accessibility. By combining advanced object detection capabilities with text-to-speech functionality, this project aims to create a scalable solution for aiding visually impaired individuals in navigating their surroundings more effectively. Furthermore, the system leverages high-performance hardware to ensure real-time operation and reliability.Hardware:


The hardware setup consists of a standard webcam that captures the live video feed and a PC equipped with a GPU. The GPU is essential for accelerating the inference process, especially for deep learning models like YOLOv11, which require significant computational power for real-time performance. The webcam captures video in real-time, and the GPU ensures that the object detection model processes each frame efficiently without introducing latency.

Software:
The software stack for the project includes Python 3.9, along with libraries and frameworks for object detection and speech synthesis:

YOLO Framework: Used for object detection. The pre-trained YOLOv11 model is loaded and applied to process video frames.

Pyttsx3: A Python library for offline Text-to-Speech synthesis. It is used to convert the detected object labels into audible speech.

OpenCV: Used for capturing video feed and processing images. It provides the necessary tools to manipulate and display frames in real-time.

Streamlit: Used to create the interactive user interface for real-time display of results and control of the system.
Streamlit Commands:

Running the Streamlit App: After ensuring that the necessary libraries and dependencies are installed, you can run the Streamlit app using the following commands:

Install Streamlit via pip:

bash
Copy code
pip install streamlit
Navigate to the directory where your Streamlit app (Python script) is saved.

Run the Streamlit app:

bash
Copy code
streamlit run your_app_name.py
Replace your_app_name.py with the name of your Python file containing the Streamlit code.

Once the app is running, a URL (typically http://localhost:8501) will appear in the terminal, which can be opened in a browser to access the live GUI.
