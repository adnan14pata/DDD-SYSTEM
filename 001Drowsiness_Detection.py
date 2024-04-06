from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import tkinter as tk
from PIL import Image, ImageTk

mixer.init()
mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
is_detecting = False

# Create GUI window
root = tk.Tk()
root.title("Driver Drowsiness Detection")
root.geometry("600x500")

# Create canvas to display video feed
canvas = tk.Canvas(root, width=500, height=400, bg="black")
canvas.pack(pady=20)

def detect_drowsiness():
    global flag
    if is_detecting:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                flag = 0

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        canvas.img = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        if is_detecting:
            canvas.after(10, detect_drowsiness)

def start_detection():
    global is_detecting
    if not is_detecting:
        is_detecting = True
        detect_drowsiness()

def stop_detection():
    global is_detecting
    if is_detecting:
        is_detecting = False

# Create Start and Stop buttons
start_button = tk.Button(root, text="Start Detection", command=start_detection, bg="green", fg="white", font=("Helvetica", 12, "bold"))
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white", font=("Helvetica", 12, "bold"))
stop_button.pack(pady=10)

# Exit GUI window when 'q' key is pressed
def exit_app(event):
    if event.char == 'q':
        root.destroy()

root.bind('<Key>', exit_app)

# Run the GUI main loop
root.mainloop()

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
