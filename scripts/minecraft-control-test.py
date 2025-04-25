import cv2 as cv
import mediapipe as mp
import numpy as np
import mouse
import keyboard
import time

# Some Utility Tools
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def calculate_dist(a, b):
    a = np.array(a)  # First
    b = np.array(b)  # End

    dist = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    return dist


# Webcam Init
cap = cv.VideoCapture(0)

# Mediapipe Init
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Pose Detection With Mouse and Keyboard Event
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass
        
        # Nose
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

        # Left Part
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Right Part
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # cv.putText(image, str(calculate_angle(left_shoulder, left_elbow, left_wrist)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)
        # cv.putText(image, str(calculate_dist(left_shoulder, left_elbow)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv.LINE_AA)

        # Calculate Distance and Angle
        right_ear_to_nose = calculate_dist(right_ear, nose)
        left_ear_to_nose = calculate_dist(left_ear, nose)
        right_hand_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_hand_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_hip_to_ankle = calculate_dist(right_hip, right_ankle)
        left_hip_to_ankle = calculate_dist(left_hip, left_ankle)
        ankle_to_ankle = calculate_dist(left_ankle, right_ankle)
        ear_to_shoulder = calculate_dist(left_ear, left_shoulder)
        

        # Event Condition (Look) == Prob (Raw Input Off, Solved!)
        # print(right_ear_to_nose)
        if right_ear_to_nose < 0.02:
            cv.putText(image, "Look Right!", (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            mouse.move(10, 0, absolute=False, duration=0.2)
            # mouse.move(760, 540)
        elif left_ear_to_nose < 0.02:
            cv.putText(image, "Look Left!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            mouse.move(-10, 0, absolute=False, duration=0.2)
            # mouse.move(1160, 540)
        # elif ear_to_shoulder < 1.2:
        #     cv.putText(image, "Look Down!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        #     mouse.move(0, 10, absolute=False, duration=0.2)
        # elif ear_to_shoulder > 1.2:
        #     cv.putText(image, "Look Up!", (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        #     mouse.move(0, -10, absolute=False, duration=0.2)
        else:
            cv.putText(image, "Look Forward!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(image, "Look Forward!", (400, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            # mouse.move(960, 540)
        
        # Event Condition (Hit)
        if right_hand_angle < 90:
            cv.putText(image, "Right Hand!", (400, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            mouse.click('left')
        else:
            cv.putText(image, "Not Clicked!", (400, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        # Event Condition (Put)
        if left_hand_angle < 90:
            cv.putText(image, "Left Hand!", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            mouse.click('right')
        else:
            cv.putText(image, "Not Clicked!", (50, 75), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        
        # Event Condition (Shift)
        if right_hip_to_ankle < 0.35 and left_hip_to_ankle < 0.35:
            cv.putText(image, "Shift Pressed!", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            keyboard.press('shift')
        # else:
        #     cv.putText(image, "Shift Released!", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        # Event Condition (Jump)
        # print(right_shoulder[1])
        elif right_shoulder[1] < 0.2 and left_shoulder[1] < 0.2:
            cv.putText(image, "Space Pressed!", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            keyboard.press('space')
        else:
            cv.putText(image, "Space/Shift Released!", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            keyboard.release('shift')
            keyboard.release('space')

        # Event Condition (Walking) == Prob (Raw Input Off, Solved!)
        if ankle_to_ankle < 0.1:
            cv.putText(image, "Walking!", (50, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            keyboard.press('w')
        else:
            cv.putText(image, "Stop!", (50, 125), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
            keyboard.release('w')
            


            

            

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv.imshow('Webcam', frame)
        cv.imshow('Mediapipe Feed', image)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()