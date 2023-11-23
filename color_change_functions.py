import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math
from math import sqrt


# ********** FIND CIRCLE ********** 
# Function to find the circle on 
# which the given three points lie 

#TODO program currenlty has no way of handling errors due to division by 0. 
#TODO this happens when the input points are too close together 

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
    r = round(sqrt(sqr_of_r), 5); 
 
    return r, h, k 


# ********** GET RADIUS AND ANGLE ********** 
#this function returns the radius of the circle (realtive to the hand size)
# and the angle of the line formed between the middle finder and thumb relative to the vertial Y-axis 

def getRadiusAndAngle(results, image, height, width, mp_hands, display=False):
    result_angle = float("inf")
    result_rad = -1
    if results.multi_hand_landmarks: #results.multi_handf_landmarks is the results array
        for num, hand in enumerate(results.multi_hand_landmarks): 
            #extract finger tip landmarks
            thum_coords = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_coords = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_coords = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            #calculate the slope of the line formed by the thumb and middle finger
            thumb_middle_slope = (thum_coords.y-middle_coords.y) / (thum_coords.x - middle_coords.x)        
            angle = np.arctan(thumb_middle_slope) #returns the value between [-pi/2, pi/2]. unit Radians
            # print("angle: ", angle)
            result_angle = angle + np.pi/2  
            #TODO figure out how to enforce bounds for the angle calculation. Atan only returns between a range [-pi/2, pi/2]. We need to enforce this some how or figure out how to get full 360 degrees of rotation 
            
            #denoirmalize the coords of the finger tips
            denorm_thumb_coords = (int(thum_coords.x*width), int(thum_coords.y*height))
            denorm_middle_coords = (int(middle_coords.x*width), int(middle_coords.y*height))
            denorm_index_coords = (int(index_coords.x*width), int(index_coords.y*height))

            #extract the distance from the writst to the index MCP and use as a reference to normalize the circle value
            wrist_coords = hand.landmark[mp_hands.HandLandmark.WRIST]
            indexmcp_coords = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            wrist_dist = math.dist([wrist_coords.x, wrist_coords.y], [indexmcp_coords.x, indexmcp_coords.y])

            #calcualte circle formed by three points 
            r, h, k = findCircle(denorm_thumb_coords[0], denorm_thumb_coords[1], denorm_index_coords[0], denorm_index_coords[1], denorm_middle_coords[0], denorm_middle_coords[1])
            
            #radius value relative to hand size 
            result_rad = r/wrist_dist
            #TODO (optional) currently im resizing the radius of the circle by dividing by the segment formed between the wrist and the base of the index finger.
                #this is done to try to keep the circle radius independent of the distance the hand is to the webcam 
                #this kind of works right now but there might be better ways to do this 
            

            #display attirbutes on video feed
            if display:
                #draw the triangle formed by the thumb middle and index
                cv2.line(image, denorm_index_coords, denorm_thumb_coords, (255, 255, 255), 2)
                cv2.line(image, denorm_thumb_coords, denorm_middle_coords, (255, 255, 255), 2)
                cv2.line(image, denorm_middle_coords, denorm_index_coords, (255, 255, 255), 2)

                #draw circumscribed circle about the three finger tips 
                cv2.circle(image, (int(h),int(k)), int(r), (0, 255, 0), 2) #cast radius and center coordinates as an interger

                #put angle and circle radius on the image 
                cv2.putText(image, "adjusted radius: " + str(result_rad), (80,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(image, "adjusted angle (0, pi): " + str(result_angle), (80,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return result_rad, result_angle


# ********** DISPLAY ALL LANDMARKS ********** 
# takes tracking result and displays nodes at each joints and line segments between joints 
def display_all_landmarks(results, image, mp_hands):
    if results.multi_hand_landmarks: #results.multi_handf_landmarks is the results array
        for num, hand in enumerate(results.multi_hand_landmarks): 
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                    #pass in image, hand, and hand_connections
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                    )




#based off: https://github.com/nicknochnack/MediaPipeHandPose
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0) #initialize video feed
external_results = 0

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 1) as hands: #instantiate mediapipe hands model
    #set min_detection_confidence = 80%
        #initially we want ouir detection confidecnt to eb 80-%
    #set min-tracking_confidence = 50%
        #once the hand is detected then  the threshold is 50%

    while cap.isOpened(): #while video feed is active
        ret, frame = cap.read() #by default the image feed is in BGR form 
        # print("image dimensions: ", frame.shape[:2])

        #extract the height and width of the frame
        height, width = frame.shape[:2]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #recolor frame to RGB
        # image = cv2.flip(image, 1) # Flip on horizontal

        image.flags.writeable = False # Set flag
        results = hands.process(image)# this is what is actually doing the detections 
        image.flags.writeable = True # Set flag to true

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #recolor back to to BGR
        
        rad, angle =  getRadiusAndAngle(results, image, height, width, mp_hands, display=True)

        norm_angle = angle/np.pi #normalize so the output is [0, 1]
        norm_rad = rad/1200 #normalize using 1200 value as the upper radius. experimentally determined 

        print("norm_rad: ", norm_rad)
        print("angle_rad: ", norm_angle)

        test_color = (norm_angle*255*norm_rad, norm_rad*255, norm_rad*255)
        print(test_color)

        cv2.circle(image, (1700, 160), 80, test_color, 9)


        external_results = results
        cv2.imshow('Hand Tracking', image) #show the inmage which has the rendering 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
