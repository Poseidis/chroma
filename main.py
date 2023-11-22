import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import draw_landmarks_on_image

model_path = "model/hand_landmarker.task"
# model from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE) # change to VIDEO

detector = HandLandmarker.create_from_options(options)

test_img = mp.Image.create_from_file("test/test.jpg")

out = detector.detect(test_img)
# print(out.hand_landmarks)

annotated_image = draw_landmarks_on_image(test_img.numpy_view(), out)
cv2.imshow("test", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))