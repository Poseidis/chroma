from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import math

## VISUALIZATION CODE FROM https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(annotated_image,
                                            hand_landmarks_proto,
                                            solutions.hands.HAND_CONNECTIONS,
                                            solutions.drawing_styles.get_default_hand_landmarks_style(),
                                            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def findCircle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2; 
    x13 = x1 - x3; 

    y12 = y1 - y2; 
    y13 = y1 - y3; 

    y31 = y3 - y1; 
    y21 = y2 - y1; 

    x31 = x3 - x1; 
    x21 = x2 - x1; 

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2); 

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2); 

    sx21 = pow(x2, 2) - pow(x1, 2); 
    sy21 = pow(y2, 2) - pow(y1, 2); 

    f = (((sx13) * (x12) + (sy13) *
          (x12) + (sx21) * (x13) +
          (sy21) * (x13)) // (2 *
          ((y31) * (x12) - (y21) * (x13))));
              
    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
          (2 * ((x31) * (y12) - (x21) * (y13)))); 

    c = (-pow(x1, 2) - pow(y1, 2) -
          2 * g * x1 - 2 * f * y1); 

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g; 
    k = -f; 
    sqr_of_r = h * h + k * k - c; 

    # r is the radius 
    r = round(math.sqrt(sqr_of_r), 5); 

    return r, h, k 


def getRadTheta(results, height, width):
    for landmark in results.hand_landmarks:
        #extract finger tip landmarks
        thum_coords = landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_coords = landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_coords = landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        #calculate the slope of the line formed by the thumb and middle finger
        thumb_middle_slope = (thum_coords.y-middle_coords.y) / (thum_coords.x - middle_coords.x)        
        angle = np.arctan(thumb_middle_slope)
        result_angle = angle + np.pi/2
        denorm_thumb_coords = (int(thum_coords.x*width), int(thum_coords.y*height))
        denorm_middle_coords = (int(middle_coords.x*width), int(middle_coords.y*height))
        denorm_index_coords = (int(index_coords.x*width), int(index_coords.y*height))

        #extract the distance from the writst to the index MCP and use as a reference to normalize the circle value
        wrist_coords = landmark[mp.solutions.hands.HandLandmark.WRIST]
        indexmcp_coords = landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        wrist_dist = math.dist([wrist_coords.x, wrist_coords.y], [indexmcp_coords.x, indexmcp_coords.y])

        #calcualte circle formed by three points 
        r, h, k = findCircle(denorm_thumb_coords[0], denorm_thumb_coords[1], denorm_index_coords[0], denorm_index_coords[1], denorm_middle_coords[0], denorm_middle_coords[1])
        
        #radius value relative to hand size 
        result_rad = r/wrist_dist
    
    return result_rad, result_angle, (r, h, k)