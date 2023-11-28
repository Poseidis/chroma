import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils
from PIL import Image
import math

# model from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
MODEL_PATH = "model/hand_landmarker.task"

RING_PINKY_THRESH = 0.1 # threshold for ring and pinky finger distance to wrist/MCPs to be considered touching
RADIUS_NORMALIZATION = 650 # radius of circle to normalize to. experimentally determined
COLOR_QUANTIZATION = 18 # number of colors to quantize the color wheel into
BRIGHTNESS_QUANTIZATION = 10 # number of brightness levels to quantize the brightness into

# MediaPipe Hand Landmarker guide: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# global hand_result
# hand_result = HandLandmarkerResult()

class LiveHands():
    def __init__(self):
        self.annotated_frame = np.zeros((640,480,3), np.uint8)
        self.circle_frame = np.zeros((640,480,3), np.uint8)
        self.hand_res = None
        options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL_PATH),
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=1,
                                        min_hand_detection_confidence=0.2,
                                        min_hand_presence_confidence=0.2,
                                        min_tracking_confidence=0.3,
                                        result_callback=self.results_cb)
    
        self.detector = HandLandmarker.create_from_options(options)

    def results_cb(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        # draw the landmarks on the image
        annotated_frame = output_image.numpy_view()
        annotated_frame = utils.draw_landmarks_on_image(annotated_frame, result)

        # brightness/color
        try:
            if result is not None:
                if len(result.hand_world_landmarks) > 0:
                    d4, d5 = utils.getDistanceRingPinkyWrist(result)
                    if (d4 + d5)/2 < RING_PINKY_THRESH:
                        # print(result.hand_world_landmarks)
                        rad, angle, circle = utils.getRadTheta(result, output_image.height, output_image.width)

                        # normalize so the output is [0, 1], quantized
                        norm_angle = round(angle/np.pi*COLOR_QUANTIZATION)/COLOR_QUANTIZATION

                        # normalize using 1200 value as the upper radius quantized
                        norm_rad = round(rad/RADIUS_NORMALIZATION*BRIGHTNESS_QUANTIZATION)/BRIGHTNESS_QUANTIZATION
                        if norm_rad > 1: norm_rad = 1

                        # print("norm_rad: ", norm_rad)
                        # print("angle_rad: ", norm_angle)

                        # color = utils.getRGBFromAngleBrightness(norm_angle, norm_rad)
                        color = utils.getRGBFromAngle(norm_angle)

                        # show circle in center of new window
                        circle_frame = np.zeros((output_image.height, output_image.width, 3), np.uint8)


                        # cv2.circle(annotated_frame, (circle[1], circle[2]), int(circle[0]*norm_rad), color, -1)
                        cv2.circle(circle_frame, (output_image.width//2, output_image.height//2), int(output_image.height*.85*norm_rad), color, -1)

                        self.circle_frame = circle_frame
            self.annotated_frame = annotated_frame
        except Exception as e:
            print(e)
            pass

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # height, width = frame.shape[:2]
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            self.detector.detect_async(mp_image, frame_timestamp_ms)

            cv2.imshow("camera", self.annotated_frame) # cv2.cvtColor(self.annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("circle", self.circle_frame)
            # END

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_landmarker = LiveHands()
    hand_landmarker.run()