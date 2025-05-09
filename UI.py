import av
import cv2
import os
import streamlit as st
import numpy as np
from detection.utils import drawBoxes
from detection.model import getModel, getPrediction
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes


st.title("Simple UI for Face Mask Detection")
os.makedirs('camera', exist_ok=True) # Directory to store images

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    cv2.imwrite('camera/realtime.png', img)
    predictedBoxes = getPrediction(model, 'camera/realtime.png')
    predictedImage = drawBoxes(img, predictedBoxes, withConfScore=True, isRgb=False)
    return av.VideoFrame.from_ndarray(predictedImage, format="bgr24")


# Upload model's weights
os.makedirs('uploaded-weights', exist_ok=True)
uploadWeightsChkbox = st.checkbox('Upload new weights?')
if uploadWeightsChkbox:
    uploadedWeight = st.file_uploader("Upload the file of model's weights (.pt)")
    if uploadedWeight is not None:
        with open(f"uploaded-weights/weights.pt", mode='wb') as f:
            f.write(uploadedWeight.getbuffer())

pretrainedWeight = 'uploaded-weights/weights.pt'
if not os.path.exists(pretrainedWeight):
    st.error('Please upload an weights to continue!')
else:
    model = getModel(pretrainedWeight)
    st.success('Success')
    on = st.toggle("Using real-time camera")

    if on:
        webrtc_streamer(
            key="example", 
            video_frame_callback=video_frame_callback,
            video_html_attrs=VideoHTMLAttributes(
            autoPlay=True, controls=True, style={"width": "100%"}, muted=True),
        )
        
    else:
        uploaded_file = st.file_uploader('Upload image to predict')
        cols = st.columns(2)

        if uploaded_file is not None:
            # Read the image and reshape 
            uploaded_image = Image.open(uploaded_file).convert('RGB')
            resize_shape = (480, 480 * uploaded_image.size[1] // uploaded_image.size[0])
            uploaded_image = uploaded_image.resize(resize_shape)
            uploaded_image.save('camera/uploaded_image.png')
            uploaded_image = np.array(uploaded_image)
            
            # Get the prediction bounding box and draw on image
            predicted_boxes = getPrediction(model, 'camera/uploaded_image.png')
            predicted_image = drawBoxes(uploaded_image, predicted_boxes, withConfScore=True)
            
            cols[0].image(uploaded_image, caption="Original image")
            cols[1].image(predicted_image, caption="Predicted image")