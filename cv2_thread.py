import sys
import traceback
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage
import mediapipe as mp
from body import BodyState
from body.const import IMAGE_HEIGHT, IMAGE_WIDTH
import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pydirectinput
import math
import subprocess

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BG_COLOR = (192, 192, 192)  # gray

# Path to the AutoHotkey script
ahk_script = r'script.ahk'

# Path to the AutoHotkey executable
ahk_exe = r"C:\Program Files\AutoHotkey\AutoHotkey.exe"  # Update this path with the correct one

def get_label(index, hand, results, mp_hands):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output


class Cv2Thread(QThread):
    update_frame = Signal(QImage)
    update_state = Signal(dict)

    def __init__(
        self, shooting_mode=True, parent=None, mp_config=None, body_config=None, events_config=None
    ):
        QThread.__init__(self, parent)
        self.status = True
        self.cap = True
        self.body = BodyState(body_config, events_config)
        self.mp_config = mp_config
        self.shooting_mode = shooting_mode ################################ MAKE IT FALSE for RPG

    def run(self):
        print("run mediapipe", self.mp_config)
        self.cap = cv2.VideoCapture(0)

        if self.shooting_mode:
            # Initialize MediaPipe Hands
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(max_num_hands=2)
            mp_drawing = mp.solutions.drawing_utils

            # Webcam dimensions
            wCam, hCam = 640, 480

            # Frame reduction and smoothening parameters
            frameR = 150
            smoothening = 1

            # Initialize pydirectinput for mouse control
            wScr, hScr = 2560,1600 # pydirectinput.size()

            # Initialize variables for mouse movement
            plocX, plocY = 0, 0
            clocX, clocY = 0, 0
            pTime = time.time()  # Initialize pTime

            # Start video capture
            cap = cv2.VideoCapture(0)
            cap.set(3, wCam)
            cap.set(4, hCam)

        with mp_pose.Pose(**self.mp_config) as pose:
            while self.cap.isOpened() and self.status:
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                if self.shooting_mode:
                    
                    # Convert the image to RGB for processing with MediaPipe
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process the image with MediaPipe Hands
                    results = hands.process(img_rgb)

                     # Draw landmarks and annotations on the frame
                    if results.multi_hand_landmarks:

                        for hand_num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            # Draw hand landmarks and connections
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            exists = get_label(hand_num, hand_landmarks, results, mp_hands)
                            if exists:
                                text, coord = exists[0],exists[1]
                                cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                                if text.split(' ')[0]=="Left":
                                    # Get the coordinates of the wrist
                                    x_wrist, y_wrist = hand_landmarks.landmark[8].x * wCam, hand_landmarks.landmark[8].y * hCam

                                    # Map wrist coordinates to screen coordinates
                                    x_mapped = np.interp(x_wrist, (frameR, wCam - frameR), (0, wScr))
                                    y_mapped = np.interp(y_wrist, (frameR, hCam - frameR), (0, hScr))

                                    # Smoothen the values
                                    clocX = plocX + (x_mapped - plocX) / smoothening
                                    clocY = plocY + (y_mapped - plocY) / smoothening

                                    # Move the mouse cursor
                                    # pydirectinput.moveTo(int(wScr - clocX), int(clocY))
                                    input_coordinates = (int(wScr - clocX), int(clocY))  # Example coordinates

                                    # Start the AutoHotkey script using subprocess
                                    process = subprocess.Popen([ahk_exe, ahk_script, \
                                                                str(input_coordinates[0]), str(input_coordinates[1])], shell=True)


                                    # Update previous location
                                    plocX, plocY = clocX, clocY

                                else:
                                                #click if right hand 
                                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                                    # Get the index finger tip landmark
                                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                                    # Calculate the distance between thumb tip and index finger tip
                                    distance = math.sqrt((thumb_tip.x - index_finger_tip.x) ** 2 + (thumb_tip.y - index_finger_tip.y) ** 2 + (thumb_tip.z - index_finger_tip.z) ** 2)

                                    if distance < 0.03:
                                        pydirectinput.mouseDown(button= "left")
                                    else:
                                        pydirectinput.mouseUp(button="left")

        

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                # Recolor image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if (
                    self.mp_config["enable_segmentation"]
                    and results.segmentation_mask is not None
                ):
                    try:
                        # Draw selfie segmentation on the background image.
                        # To improve segmentation around boundaries, consider applying a joint
                        # bilateral filter to "results.segmentation_mask" with "image".
                        condition = (
                            np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                        )
                        # The background can be customized.
                        #   a) Load an image (with the same width and height of the input image) to
                        #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
                        #   b) Blur the input image by applying image filtering, e.g.,
                        #      bg_image = cv2.GaussianBlur(image,(55,55),0)
                        bg_image = cv2.GaussianBlur(image, (55, 55), 0)
                        if bg_image is None:
                            bg_image = np.zeros(image.shape, dtype=np.uint8)
                            bg_image[:] = BG_COLOR
                        image = np.where(condition, image, bg_image)
                    except Exception:
                        print(traceback.format_exc())

                # Draw landmark annotation on the image.
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                self.body.calculate(image, results)

                # Reading the image in RGB to display it
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Creating and scaling QImage
                h, w, ch = image.shape
                image = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
                image = image.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, Qt.KeepAspectRatio)

                # Emit signal
                self.update_frame.emit(image)
                self.update_state.emit(dict(body=self.body))

                if cv2.waitKey(5) & 0xFF == 27:
                    break

        sys.exit(-1)
