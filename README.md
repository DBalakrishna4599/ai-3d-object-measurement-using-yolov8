# AI 3D Object Measurement System

This project provides a powerful tool for performing 3D measurements of objects using a standard webcam or a local IP camera. The application runs on your local machine, allowing it to directly connect to cameras on your private network.

## ✨ Features

- **Local Execution:** Runs securely on your own machine, ensuring access to local network devices.
- **AI-Powered Detection:** Utilizes the YOLOv8 model for fast and accurate object detection.
- **Flexible Camera Support:** Works with both built-in webcams (camera index 0) and local IP camera streams.
- **3D Calculations:** Estimates the depth, width, and height of detected objects in centimeters.
- **Clean Architecture:** Backend AI logic is fully separated from the Streamlit UI.

---

## 🛠️ Setup & Installation

Follow these steps to set up and run the project on your computer.

### 1. Prerequisites

- Python 3.9+ installed on your system.
- Git installed on your system.

### 2. Clone the Repository

Open your terminal or command prompt and run the following commands:

```

git clone https://github.com/DBalakrishna4599/ai-3d-object-measurement-using-yolov8.git
```
```
cd ai-3d-object-measurement-using-yolov8
```

### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

Create the environment:

```
python3 -m venv venv

```

Activate the environment:

```
#MacOS/Linux:

source venv/bin/activate
```

```
#On Windows:

venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```
pip install -r requirements.txt

```



### 5. Download the YOLOv8 Model

The first time you run the application, it will automatically download the `yolov8m.pt` model file (approx. 50 MB) into the project directory. Please ensure you have an internet connection for this first run.

---

## 🚀 How to Run the Application

1.  Ensure your virtual environment is activated and you are in the project's root directory.
2.  Run the Streamlit application with the following command:
    ```
    streamlit run frontend.py
    ```
3.  Your web browser will open with the application running at a local address (e.g., `http://localhost:8501`).
4.  Follow the instructions in the app's sidebar to connect to your local webcam or enter the address of your local IP camera.


