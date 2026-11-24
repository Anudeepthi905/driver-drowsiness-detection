"""
Driver Drowsiness Detection
===========================

Real-time system that watches a driver's eyes through a webcam and sounds an
alarm when the eyes stay closed for too long.

Pipeline for every video frame:
    1. Capture a frame from the webcam.
    2. Detect the face (Dlib HOG detector) and its 68 facial landmarks
       (Dlib shape predictor).
    3. Compute the Eye Aspect Ratio (EAR) for each eye from its 6 landmarks.
    4. Classify the eyes as OPEN or CLOSED by comparing EAR to a threshold.
    5. Count consecutive CLOSED frames; a short run is a blink, a long run
       is drowsiness.
    6. Start the alarm when the count passes the limit; stop it when the
       eyes reopen.

Run:   python drowsiness_detector.py
Quit:  press 'q' in the video window.
"""

import os
import sys

# Hide pygame's "Hello from the pygame community" banner (cosmetic only).
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2
import dlib
import imutils
import pygame
from imutils import face_utils
from scipy.spatial import distance as dist

# --------------------------------------------------------------------------
# Configuration constants (tune these for your own face, camera and lighting)
# --------------------------------------------------------------------------

# Folder containing this script, so files are found no matter where you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EAR_THRESHOLD = 0.25        # EAR below this counts as "closed": raise it if real closures are missed, lower it if open eyes read as closed.
CONSEC_FRAMES_LIMIT = 48    # Closed frames in a row needed to trigger the alarm (about 2 s at 24 fps): higher means fewer false alarms but a slower response.
CAMERA_INDEX = 0            # Webcam to open: 0 is the built-in/default camera, try 1 or 2 for an external USB camera.
ALARM_SOUND_FILE = os.path.join(BASE_DIR, "sounds", "alarm.wav")  # Sound looped while the alarm is active.

MODEL_FILE = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")  # Dlib 68-point landmark model (download separately, see README).
FRAME_WIDTH = 450           # Frames are shrunk to this width before detection: smaller is faster, but faces far from the camera may be missed.

# Landmark index ranges for each eye in the 68-point model
# (left eye = points 42-47, right eye = points 36-41, from the subject's point of view).
(L_START, L_END) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_START, R_END) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Overlay colours in OpenCV's BGR order.
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


# --------------------------------------------------------------------------
# Setup: model, alarm and camera
# --------------------------------------------------------------------------

def load_models(model_path):
    """Load Dlib's face detector and the 68-point landmark predictor.

    Exits with a helpful message if the landmark model file is missing,
    because the program cannot do anything useful without it.
    """
    if not os.path.isfile(model_path):
        sys.exit(
            f"[ERROR] Landmark model not found at:\n  {model_path}\n"
            "Download shape_predictor_68_face_landmarks.dat.bz2 from\n"
            "  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
            "extract it, and place the .dat file in the 'models' folder (see README)."
        )
    detector = dlib.get_frontal_face_detector()    # HOG + linear SVM face detector (fast on a CPU)
    predictor = dlib.shape_predictor(model_path)   # Regression-tree model that places 68 landmarks
    return detector, predictor


def init_alarm(sound_path):
    """Load the alarm sound with pygame's mixer.

    Choice: if the sound file is missing or no audio device is available, we
    warn and keep running with the on-screen alert only, instead of quitting.
    Detection still works, and the problem is reported clearly.
    Returns a pygame Sound object, or None if sound is unavailable.
    """
    if not os.path.isfile(sound_path):
        print(f"[WARNING] Alarm sound not found at: {sound_path}\n"
              "          Continuing with the visual alert only.")
        return None
    try:
        pygame.mixer.init()
        return pygame.mixer.Sound(sound_path)
    except pygame.error as exc:
        print(f"[WARNING] Could not initialise audio ({exc}).\n"
              "          Continuing with the visual alert only.")
        return None


def open_camera(index):
    """Open the webcam, or exit with a clear message if it is unavailable."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Could not open webcam at index {index}.\n"
            "Check that a camera is connected, that no other app (Zoom, Teams, ...) "
            "is using it, and that camera access is allowed in your OS privacy "
            "settings. You can also try another CAMERA_INDEX (1, 2, ...)."
        )
    return cap


# --------------------------------------------------------------------------
# Per-frame processing
# --------------------------------------------------------------------------

def capture_frame(cap):
    """Read one frame and shrink it to FRAME_WIDTH for faster processing.

    Returns the resized frame, or None if the camera stopped delivering frames.
    """
    ok, frame = cap.read()
    if not ok:
        return None
    return imutils.resize(frame, width=FRAME_WIDTH)  # keeps the aspect ratio


def detect_landmarks(gray, detector, predictor):
    """Find the face and return its 68 landmarks as a (68, 2) NumPy array.

    Choice: if several faces are visible we use the largest one, which is
    almost always the driver (closest to the camera).
    Returns None if no face is found.
    """
    # The second argument (0) means "do not upsample the image": faster, and
    # enough for a face close to a webcam.
    faces = detector(gray, 0)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, face)
    return face_utils.shape_to_np(shape)  # convert to an array of (x, y) points


def eye_aspect_ratio(eye):
    """Compute the Eye Aspect Ratio (EAR) of one eye from its 6 landmarks.

    Landmarks p1..p6 go around the eye: p1 and p4 are the corners,
    p2/p3 lie on the upper lid, and p6/p5 lie on the lower lid.

              ||p2 - p6|| + ||p3 - p5||
        EAR = -------------------------
                  2 * ||p1 - p4||

    The top holds the two vertical lid-to-lid distances; the bottom holds the
    horizontal eye width (doubled, because there are two vertical
    distances). When the eye closes the vertical distances shrink towards 0
    while the width stays almost the same, so EAR falls sharply.
    """
    a = dist.euclidean(eye[1], eye[5])  # vertical distance ||p2 - p6||
    b = dist.euclidean(eye[2], eye[4])  # vertical distance ||p3 - p5||
    c = dist.euclidean(eye[0], eye[3])  # horizontal distance ||p1 - p4||
    return (a + b) / (2.0 * c)


def compute_ear(landmarks):
    """Return the average EAR of both eyes, plus each eye's landmark points.

    Choice: we average the two eyes. Both eyes blink together, and averaging
    reduces the noise of a single slightly mis-placed landmark.
    """
    left_eye = landmarks[L_START:L_END]
    right_eye = landmarks[R_START:R_END]
    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    return ear, left_eye, right_eye


def classify_eye_state(ear, threshold=EAR_THRESHOLD):
    """Label the eyes OPEN or CLOSED by comparing EAR to the threshold."""
    return "CLOSED" if ear < threshold else "OPEN"


def update_closed_counter(counter, eye_state):
    """Count consecutive CLOSED frames; any OPEN frame resets the count to 0.

    A blink lasts only a few frames, so the counter never gets far before it
    is reset. Drowsiness keeps the eyes shut, so the counter keeps climbing.
    """
    return counter + 1 if eye_state == "CLOSED" else 0


def is_drowsy(counter, limit=CONSEC_FRAMES_LIMIT):
    """The driver counts as drowsy once the eyes have been closed for `limit` frames in a row."""
    return counter >= limit


# --------------------------------------------------------------------------
# Alarm control
# --------------------------------------------------------------------------

def update_alarm(sound, should_sound, alarm_on):
    """Start or stop the alarm only when its state needs to change.

    `alarm_on` remembers whether the sound is already playing, so we do not
    restart it on every frame. Returns the new alarm state.
    """
    if should_sound and not alarm_on:
        if sound is not None:
            sound.play(loops=-1)  # -1 = loop until stopped
        return True
    if not should_sound and alarm_on:
        if sound is not None:
            sound.stop()
        return False
    return alarm_on


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def draw_overlay(frame, ear, eye_state, counter, drowsy, left_eye=None, right_eye=None):
    """Draw eye outlines, the live EAR, the eye state, and the alert message."""
    if ear is None:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
        return

    colour = RED if eye_state == "CLOSED" else GREEN

    # Outline each eye with the convex hull of its 6 landmarks.
    for eye in (left_eye, right_eye):
        cv2.drawContours(frame, [cv2.convexHull(eye)], -1, colour, 1)

    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"Eyes: {eye_state}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    # Showing the counter makes it easy to check the frame limit during testing.
    cv2.putText(frame, f"Closed frames: {counter}/{CONSEC_FRAMES_LIMIT}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    if drowsy:
        cv2.putText(frame, "DROWSINESS ALERT!", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    detector, predictor = load_models(MODEL_FILE)
    alarm_sound = init_alarm(ALARM_SOUND_FILE)
    cap = open_camera(CAMERA_INDEX)

    closed_counter = 0   # consecutive frames with eyes closed
    alarm_on = False     # is the alarm currently sounding?

    print("[INFO] Running. Press 'q' in the video window to quit.")
    try:
        while True:
            frame = capture_frame(cap)
            if frame is None:
                print("[ERROR] Lost the webcam feed. Stopping.")
                break

            # Dlib's detector and predictor work on a single-channel image.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = detect_landmarks(gray, detector, predictor)

            if landmarks is None:
                # Choice: with no face we cannot measure the eyes, so we reset
                # the counter and silence the alarm rather than guess.
                # (Limitation: a head that drops out of view is not flagged.)
                closed_counter = 0
                alarm_on = update_alarm(alarm_sound, False, alarm_on)
                draw_overlay(frame, None, None, 0, False)
            else:
                ear, left_eye, right_eye = compute_ear(landmarks)
                eye_state = classify_eye_state(ear)
                closed_counter = update_closed_counter(closed_counter, eye_state)
                drowsy = is_drowsy(closed_counter)
                alarm_on = update_alarm(alarm_sound, drowsy, alarm_on)
                draw_overlay(frame, ear, eye_state, closed_counter, drowsy,
                             left_eye, right_eye)

            cv2.imshow("Driver Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Clean shutdown, run on 'q', on an error, or on Ctrl+C.
        update_alarm(alarm_sound, False, alarm_on)
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Shut down cleanly.")


if __name__ == "__main__":
    main()
