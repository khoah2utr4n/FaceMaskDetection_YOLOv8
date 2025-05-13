# Face Mask Detection with YOLOv8
This project implements a face mask detection system using [YOLOv8](https://github.com/ultralytics/ultralytics) model from Ultralytics and the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) on Kaggle. It can classify faces into three categories: 
  
  * Wearing a mask
  
  * Wearing a mask incorrectly
  
  * Not wearing a mask

A user-friendly interface built with Streamlit and PyQt allows real-time detection via webcam or image uploads.

![Screenshot 2025-05-09 094821](https://github.com/user-attachments/assets/dc2a3b5f-992f-4a7c-916b-b5a12c1fc93c)

## Setup
### 1. Create a virtual environment 
  ```
  conda create --name myenv python
  conda activate myenv
  ```
### 2. Clone this repository and install packages
  * Clone this repository:
  ```
  git clone https://github.com/khoah2utr4n/FaceMaskDetection_YOLOv8.git
  ```
  * Install [PyTorch GPU/CPU](https://pytorch.org/get-started/locally/).
  * Install packages
  ```
  pip install -r requirements.txt
  ```
### 3. Dataset
  * Download the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection).
  * Extract it and place the `images` and `annotations` folders inside the `datasets` directory.
  * Preprocess the data:
  ```
  python preprocessing_data.py
  ```

## Usage
### 1. Training
  * Navigate to the `detection` directory::
  ```
  cd detection
  ```
  * To train the model using the YOLOv8n weights:
  ```
  python train.py --epochs 50 --weightsPath yolov8n.pt
  ```
  * To resume training from a checkpoint:
  ```
  python train.py --weightsPath <path/to/last.pt> --resume True
  ```
  After training, the best weights will be saved as `best.pt`.

### 2. Running Detection
  * Return to the project root:
  ```
  cd ..
  ```
  * Option 1: Use the Streamlit Web UI
  ```
  streamlit run UI.py
  ```
  * Option 2: Use the PyQt GUI
  ```
  python pyqt_ui.py
  ```
   * **📥 Upload Model Weights**
     * To start, please upload your own `.pt` weights file
     * Or use my pre-trained weights [download pre-trained weights](https://drive.google.com/file/d/1PJU0x9jH14CRtRaXwB7sP7WTnRtUN0T0/view?usp=sharing).
  
  * **🖼️ Detection methods**: Choose your preferred detection method:
    * **Image upload**: Upload an image to detect mask.
    * **Real-time camera**: Use your webcam for live mask detection.
