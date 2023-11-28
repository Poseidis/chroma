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

def find_circle_3d(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    O = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))/ (b1 + b2 + b3)
    return R, O


def getRadThetaProjection(results, height, width):
    for landmark in results.hand_landmarks:
        #extract finger tip landmarks
        thum_coords = landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_coords = landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_coords = landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        #calculate the slope of the line formed by the thumb and middle finger
        thumb_middle_slope = (thum_coords.y-middle_coords.y) / (thum_coords.x - middle_coords.x)        
        result_angle = np.arctan(thumb_middle_slope) + np.pi/2

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

def getRadThetaWorld(results):
    for landmark in results.hand_world_landmarks:
        #extract finger tip landmarks
        thum_coords = landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_coords = landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        middle_coords = landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

        # calculate angle of thumb-middle relative to x-axis
        thumb_middle = np.array([thum_coords.x - middle_coords.x, thum_coords.y - middle_coords.y, thum_coords.z - middle_coords.z])
        result_angle = np.arctan2(thumb_middle[1], thumb_middle[0])

        # calculate radius of circle formed by thumb, index, and middle
        r, o = find_circle_3d(np.array([thum_coords.x, thum_coords.y, thum_coords.z]), np.array([index_coords.x, index_coords.y, index_coords.z]), np.array([middle_coords.x, middle_coords.y, middle_coords.z]))
        
    return r, result_angle, o

def getDistanceRingPinkyWrist(results):
    for landmark in results.hand_landmarks:
        #extract finger tip landmarks
        ring = landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
        pinky = landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
        wrist = landmark[mp.solutions.hands.HandLandmark.WRIST]
        # ring_mcp = landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
        # pinky_mcp = landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

        # return math.dist([ring.x, ring.y], [ring_mcp.x, ring_mcp.y]), math.dist([pinky.x, pinky.y], [pinky_mcp.x, pinky_mcp.y])
        return math.dist([ring.x, ring.y], [wrist.x, wrist.y]), math.dist([wrist.x, wrist.y], [pinky.x, pinky.y])

    
def getRGBFromAngleBrightness(norm_angle, norm_rad):
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
    
    return test_color

def getRGBFromAngle(norm_angle):
    # color
    # Red = 255, 0, 0
    # Yellow = 255, 255, 0
    # Green = 0, 255, 0
    # Cyan = 0, 255, 255
    # Blue = 0, 0, 255
    # Magenta = 255, 0, 255

    # Between Red and Yellow, R stays at 255 * norm_rad, G INCREASES from 0 to 255 * norm_rad
    if 0 <= norm_angle <= 1/6:
        test_color = (255, norm_angle*1530, 0)
    # Between Yellow and Green, G stays at 255 * norm_rad, R DECREASEs from 255 * norm_rad to 0
    elif 1/6 < norm_angle <= 1/3:
        test_color = (255 - (norm_angle - 1/6)*1530, 255, 0)
    # Between Green and Cyan, G stays at 255 * norm_rad, B INCREASES from 0 to 255 * norm_rad
    elif 1/3 < norm_angle <= 1/2:
        test_color = (0, 255, (norm_angle - 1/3)*1530)
    # Between Cyan and Blue, B stays at 255 * norm_rad, G DECREASES from 255 * norm_rad to 0
    elif 1/2 < norm_angle <= 2/3:
        test_color = (0, 255 - (norm_angle - 0.5)*1530, 255)
    # Between Blue and Magenta, B stays at 255 * norm_rad, R INCREASES from 0 to 255 * norm_rad
    elif 2/3 < norm_angle <= 5/6:
        test_color = ((norm_angle - 2/3)*1530, 0, 255)
    # Between Magenta and Red, R stays at 255 * norm_rad, B DECREASES from 255 * norm_rad to 0
    elif 5/6 < norm_angle <= 1:
        test_color = (255, 0, 255 - (norm_angle - 5/6)*1530)
    else:
        test_color = (0, 0, 0)
    
    return test_color