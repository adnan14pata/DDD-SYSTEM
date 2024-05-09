import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
from pygame import mixer
from twilio.rest import Client
import json

# Initialize Pygame mixer for sound effects
mixer.init()
mixer.music.load("music.wav")

# Constants and initializations for drowsiness detection
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
is_detecting = False
alert_sent = False

# Twilio credentials
with open("config.json") as config_file:
    config = json.load(config_file)
account_sid =config["ACCOUNT_SID"] 
auth_token =config["AUTH_TOKEN"]
client = Client(account_sid, auth_token)

# Function to start drowsiness detection
def start_detection():
    global is_detecting
    if not is_detecting:
        is_detecting = True
        detect_drowsiness()  # Start drowsiness detection
        label_instruction.config(text="Monitoring driver drowsiness...")

# Function to stop drowsiness detection
def stop_detection():
    global is_detecting
    if is_detecting:
        is_detecting = False
        label_instruction.config(text="Press 'Start Detection' to begin monitoring")

# Function to send alert message
def send_alert_message():
    global alert_sent
    if not alert_sent:  # Send alert message only if it hasn't been sent already
        message = client.messages.create(
            body="ALERT: Drowsiness detected. Immediate action required!",
            from_=config["FROM_NUM"],
            to= config["TO_NUM"] 
        )
        print("Alert message sent:", message.sid)
        alert_sent = True  # Set flag to indicate that alert message has been sent

# Function to send SOS message
def send_sos_message():
    message = client.messages.create(
        body="SOS! Immediate assistance needed. Location: (18°25'13.7\"N 73°54'18.0\"E)",
        from_=config["FROM_NUM"],
        to=config["TO_NUM"] 
    )
    print("SOS message sent:", message.sid)

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect drowsiness in real-time video feed
def detect_drowsiness():
    global flag, alert_sent, is_detecting
    if is_detecting:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect(gray, 0)
        for face in faces:
            shape = predict(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[lStart:lEnd]
            right_eye = shape[rStart:rEnd]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "ALERT! Driver Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    mixer.music.play()  # Play alert sound
                    send_alert_message()  # Send alert message
            else:
                flag = 0
                alert_sent = False

        # Convert frame to RGB format and display on canvas
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        canvas.img = img  # Keep reference to prevent garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

        if is_detecting:
            canvas.after(10, detect_drowsiness)  # Call detect_drowsiness recursively

# Create main window
root = tk.Tk()
root.title("Driver Drowsiness Detection")
root.geometry("800x700")
root.configure(bg="black")  # Set window background color to black

# Create canvas for video feed display
canvas = tk.Canvas(root, width=640, height=480, bg="black")
canvas.pack(pady=20)

# Create labels for instructions
label_title = tk.Label(root, text="Driver Drowsiness Detection System", font=("Helvetica", 25, "bold"), fg="white", bg="black")
label_title.pack()

label_instruction = tk.Label(root, text="Press 'Start Detection' to begin monitoring", font=("Helvetica", 14), fg="gray", bg="black")
label_instruction.pack(pady=10)

# Create frame to contain buttons
button_frame = tk.Frame(root, bg="black")
button_frame.pack(pady=20)

# Create and style round buttons using custom class
class RoundButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(relief=tk.FLAT, overrelief=tk.RAISED, bd=4, width=10, height=2)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, event):
        self.config(bg="gray")

    def on_leave(self, event):
        self.config(bg=self['activebackground'])

# Create round buttons for Start, Stop, and SOS
start_button = RoundButton(button_frame, text="Start", command=start_detection, bg="green", fg="white", font=("Helvetica", 14, "bold"), activebackground="green3")
start_button.pack(side=tk.LEFT, padx=20)

stop_button = RoundButton(button_frame, text="Stop", command=stop_detection, bg="red", fg="white", font=("Helvetica", 14, "bold"), activebackground="red3")
stop_button.pack(side=tk.LEFT, padx=20)

sos_button = RoundButton(button_frame, text="S.O.S", command=send_sos_message, bg="blue", fg="white", font=("Helvetica", 14, "bold"), activebackground="blue3")
sos_button.pack(side=tk.LEFT, padx=20)

# Bind 'q' key to exit application
def exit_app(event):
    if event.char == 'q':
        root.destroy()

root.bind('<Key>', exit_app)

# Run the main GUI event loop
root.mainloop()

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()