import numpy as np
import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils
from PIL import Image

model_path = "model/hand_landmarker.task"
# model from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models

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
        self.hand_res = None
        options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='model/hand_landmarker.task'),
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=1,
                                        min_hand_detection_confidence=0.3,
                                        min_hand_presence_confidence=0.3,
                                        min_tracking_confidence=0.3,
                                        result_callback=self.results_cb)
    
        self.detector = HandLandmarker.create_from_options(options)

    def results_cb(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        annotated_frame = utils.draw_landmarks_on_image(output_image.numpy_view(), result)
        # self.hand_res = result

        # brightness/color
        try:
            if result is not None:
                if len(result.hand_world_landmarks) > 0:
                    rad, angle, circ = utils.getRadTheta(result, output_image.height, output_image.width)
                    norm_angle = angle/np.pi # normalize so the output is [0, 1]
                    norm_rad = rad/650 # normalize using 1200 value as the upper radius. experimentally determined 
                    if norm_rad > 1: norm_rad = 1
                    # print("norm_rad: ", norm_rad)
                    # print("angle_rad: ", norm_angle)
                    

                    # color
                    # Red = 255, 0, 0
                    # Yellow = 255, 255, 0
                    # Green = 0, 255, 0
                    # Cyan = 0, 255, 255
                    # Blue = 0, 0, 255
                    # Magenta = 255, 0, 255

                    # Between Red and Yellow, R stays at 255 * norm_rad, G INCREASES from 0 to 255 * norm_rad
                    if 0 <= norm_angle <= 1/6:
                        test_color = (norm_rad*255, norm_rad*norm_angle*1530, 0)
                    # Between Yellow and Green, G stays at 255 * norm_rad, R DECREASEs from 255 * norm_rad to 0
                    elif 1/6 < norm_angle <= 1/3:
                        test_color = (norm_rad*(255 - (norm_angle - 1/6)*1530), norm_rad*255, 0)
                    # Between Green and Cyan, G stays at 255 * norm_rad, B INCREASES from 0 to 255 * norm_rad
                    elif 1/3 < norm_angle <= 1/2:
                        test_color = (0, norm_rad*255, norm_rad*(norm_angle - 1/3)*1530)
                    # Between Cyan and Blue, B stays at 255 * norm_rad, G DECREASES from 255 * norm_rad to 0
                    elif 1/2 < norm_angle <= 2/3:
                        test_color = (0, norm_rad*(255 - (norm_angle - 0.5)*1530), norm_rad*255)
                    # Between Blue and Magenta, B stays at 255 * norm_rad, R INCREASES from 0 to 255 * norm_rad
                    elif 2/3 < norm_angle <= 5/6:
                        test_color = (norm_rad*(norm_angle - 2/3)*1530, 0, norm_rad*255)
                    # Between Magenta and Red, R stays at 255 * norm_rad, B DECREASES from 255 * norm_rad to 0
                    elif 5/6 < norm_angle <= 1:
                        test_color = (norm_rad*255, 0, norm_rad*(255 - (norm_angle - 5/6)*1530))
                    else:
                        test_color = (0, 0, 0)

                    print(test_color)
                    cv2.circle(annotated_frame, (circ[1], circ[2]), int(circ[0]), test_color, 9)
            
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

            cv2.imshow("test", self.annotated_frame) # cv2.cvtColor(self.annotated_frame, cv2.COLOR_RGB2BGR)
            # END

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    hand_landmarker = LiveHands()
    hand_landmarker.run()