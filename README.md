# AI 3D Object Measurement System

This project is a web-based application built with Streamlit and Python that performs real-time 3D measurements of objects. It uses a single camera (either a webcam or an IP camera) to capture a stereo image pair and leverages a YOLOv8 object detection model to identify objects and calculate their dimensions and depth.

## ✨ Features

- **Frontend Interface:** A user-friendly web interface built with Streamlit.
- **AI-Powered Detection:** Utilizes the YOLOv8 model for fast and accurate object detection.
- **Stereo Measurement:** Captures a left-right image pair with a timed delay to simulate a stereo camera.
- **3D Calculations:** Estimates the depth, width, and height of detected objects in centimeters.
- **Flexible Camera Support:** Works with both built-in webcams and external IP camera streams.
- **Separated Logic:** The backend (AI/CV logic) is fully separated from the frontend (UI), making the code clean and maintainable.

---

## 🛠️ Setup & Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.9+
- A GitHub account and Git installed

### 2. Clone the Repository

Clone this repository to your local machine:

```

git clone https://github.com/DBalakrishna4599/ai-3d-object-measurement-using-yolov8.git
```
```
cd ai-3d-object-measurement-using-yolov8
```

Create the environment:

```
python3 -m venv venv

```

Activate the environment:

```
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```
pip install -r requirements.txt

```


### 5. Download the YOLOv8 Model

This project requires the `yolov8m.pt` model file. Download it from [this link](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) and place it in the root of the project folder.

---

## 🚀 How to Run

1.  Ensure your virtual environment is activated and you are in the project's root directory.
2.  Run the Streamlit application with the following command:
   
    ```
    streamlit run frontend.py
    ```
3.  Your web browser will open with the application running.
4.  Follow the instructions in the "How to Use" section of the web app to connect to a camera and start a measurement.


