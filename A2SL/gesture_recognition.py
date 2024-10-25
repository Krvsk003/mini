import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def recognize_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.get_gesture(hand_landmarks)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def get_gesture(self, hand_landmarks):
        # Get fingertip and base coordinates
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]

        # Check if fingers are extended
        thumb_extended = thumb_tip.x < thumb_base.x
        index_extended = index_tip.y < index_base.y
        middle_extended = middle_tip.y < middle_base.y
        ring_extended = ring_tip.y < ring_base.y
        pinky_extended = pinky_tip.y < pinky_base.y

        # Calculate distances between fingertips
        def distance(p1, p2):
            return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

        thumb_index_close = distance(thumb_tip, index_tip) < 0.05

        # Recognize gestures
        if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "L Shape (L)"

        elif not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "Victory (V)"

        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "Open Hand (5)"

        elif not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Closed Fist (S)"

        elif not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Point (1)"

        elif thumb_index_close and not middle_extended and not ring_extended and not pinky_extended:
            return "O Shape (O)"

        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "I Love You (ILY)"

        elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "B Shape (B)"

        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Thumb Up (A)"

        else:
            return "Unknown Gesture"

    def start_recognition(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = self.recognize_gesture(frame)
            cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break

        cap.release()
        cv2.destroyAllWindows()