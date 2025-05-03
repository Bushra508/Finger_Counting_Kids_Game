import cv2
import numpy as np
import math
import random
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QStackedLayout, QGraphicsDropShadowEffect
from PyQt5.QtCore import QTimer, QPropertyAnimation, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter

# Load underwater background image
underwater = cv2.imread("dependencies\\underwater.jpg")  # Update path as needed
if underwater is None:
    underwater = np.zeros((600, 800, 3), dtype=np.uint8)
    for y in range(600):
        color = (255 - int(y/600*255), int(y/600*255), 255)
        underwater[y, :] = color
else:
    underwater = cv2.resize(underwater, (800, 600))

# Stage Questions (same as original)
STAGE1_QUESTIONS = [
    ("How many eyes do you have?", 2),
    ("How many fingers do you have on one hand?", 5),
    ("How many legs do you have?", 2),
    ("How many thumbs do you have?", 2),
    ("How many noses do you have?", 1),
    ("How many ears do you have?", 2),
    ("How many hands do you have?", 2),
    ("How many feet do you have?", 2),
    ("How many wheels does a car have?", 4),
    ("How many legs does a chair have?", 4),
    ("How many sides does a triangle have?", 3),
    ("How many colors are in a rainbow?", 7),
    ("How many days are in a week?", 7),
    ("How many letters are in the word 'cat'?", 3),
    ("How many legs does a dog have?", 4),
    ("How many wings does a bird have?", 2),
    ("How many tails does a cat have?", 1),
    ("How many wheels does a bicycle have?", 2),
    ("How many sides does a square have?", 4),
    ("How many planets are in the solar system?", 8),
    ("How many continents are there?", 7),
    ("How many oceans are there?", 5),
    ("How many colors are in a traffic light?", 3),
    ("How many fingers do you have on both hands?", 10),
    ("How many toes do you have on both feet?", 10),
    ("How many wheels does a tricycle have?", 3),
    ("How many legs does an insect have?", 6),
    ("How many points does a star (basic shape) have?", 5),
    ("How many legs does a spider have?", 8),
    ("How many eyes do most fish have?", 2),
    ("How many sides does a pentagon have?", 5),
    ("How many wheels does a unicycle have?", 1),
    ("How many wings does a butterfly have?", 4),
    ("How many legs does a frog have?", 4),
    ("How many arms does an octopus have?", 8),
    ("How many legs does a turtle have?", 4),
    ("How many legs does a horse have?", 4),
    ("How many nostrils does a human have?", 2),
    ("How many legs does a cow have?", 4),
    ("How many hooves does a goat have?", 4),
    ("How many hands does a clock have?", 2),
    ("How many teeth does a toddler usually have?", 8),
    ("How many eyes does an owl have?", 2),
    ("How many ears does a rabbit have?", 2),
    ("How many beaks does a parrot have?", 1),
    ("How many feet does a chicken have?", 2),
    ("How many toes does a dog have on one paw?", 4),
    ("How many hearts does an octopus have?", 3),
    ("How many wings does a bee have?", 4),
    ("How many horns does a rhino have?", 1),
    ("How many legs does a crab have?", 8),
    ("How many wings does a dragonfly have?", 4),
    ("How many feet does a kangaroo have?", 2),
    ("How many toes does an eagle have on each foot?", 4),
    ("How many legs does a rabbit have?", 4),
    ("How many shells does a snail have?", 1),
    ("How many tentacles does a jellyfish have?", 8),
    ("How many tusks does an elephant have?", 2),
    ("How many arms does a starfish have (most commonly)?", 5),
    ("How many feet does a flamingo stand on at a time?", 1),
    ("How many wheels does a scooter have?", 2),
    ("How many eyes does a deer have?", 2),
    ("How many hands does a gorilla have?", 2),
    ("How many flippers does a seal have?", 4),
    ("How many toes does a hippopotamus have on one foot?", 4),
    ("How many humps does a camel usually have?", 1),
    ("How many hands does an orangutan have?", 2),
    ("How many eyes does an ant have?", 2),
    ("How many legs does a cheetah have?", 4),
    ("How many hooves does a deer have?", 4),
    ("How many tails does a fox have?", 1),
    ("How many stripes does a clownfish have?", 3),
    ("How many beaks does a pelican have?", 1),
    ("How many fangs does a vampire bat have?", 2),
    ("How many wheels does a roller skate have?", 4),
    ("How many claws does a bear have on one paw?", 5),
    ("How many nostrils does a dolphin have?", 1),
    ("How many wings does a moth have?", 4),
    ("How many legs does a lizard have?", 4),
    ("How many thumbs does a panda have?", 2),
    ("How many toes does a cat have on one paw?", 4),
    ("How many wings does a mosquito have?", 2),
    ("How many eyes does a peacock have?", 2),
    ("How many feet does a pigeon have?", 2),
    ("How many pincers does a scorpion have?", 2),
    ("How many legs does a beetle have?", 6),
    ("How many thumbs does a chimpanzee have?", 2),
    ("How many wings does a cockroach have?", 4),
    ("How many fingers does a gorilla have on one hand?", 5),
    ("How many flippers does a dolphin have?", 2),
    ("How many toes does a crocodile have on one foot?", 5),
    ("How many arms does a crab have?", 2),
    ("How many fingers does a cartoon character usually have on one hand?", 4),
    ("How many eyes does a tarantula have?", 8),
    ("How many spots does a ladybug have (usually)?", 7),
    ("How many ears does an owl have?", 2),
    ("How many thumbs does a koala have?", 2),
    ("How many holes does a bowling ball have?", 3),
    ("How many front teeth do humans have?", 8),
    ("How many players are on a basketball team on the court?", 5),
    ("How many sides does a hexagon have?", 6),
    ]  

STAGE2_QUESTIONS = [("What is 2 + 3?", 5),
    ("What is 7 - 4?", 3),
    ("What is 5 + 1?", 6),
    ("What is 8 - 2?", 6),
    ("What is 1 + 6?", 7),
    ("What is 9 - 5?", 4),
    ("What is 4 + 4?", 8),
    ("What is 6 - 1?", 5),
    ("What is 3 + 3?", 6),
    ("What is 10 - 2?", 8),
    ]  

def add_shadow(widget: QWidget, blur_radius=12, x_offset=4, y_offset=4, color=QColor(0, 0, 0, 160)):
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(blur_radius)
        shadow.setXOffset(x_offset)
        shadow.setYOffset(y_offset)
        shadow.setColor(color)
        widget.setGraphicsEffect(shadow)

class SpeechThread(QThread):
    speak_signal = pyqtSignal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        engine = pyttsx3.init()
        engine.say(self.text)
        engine.runAndWait()
        self.speak_signal.emit("Speech completed")

class ConfettiParticle(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.x = random.randint(0, parent.width())
        self.y = random.randint(-50, 0)
        self.size = random.randint(5, 15)
        self.velocity = random.uniform(2, 5)
        self.setFixedSize(self.size, self.size)
        self.move(int(self.x), int(self.y))
        self.show()

    def move_down(self):
        self.y += self.velocity
        if self.y > self.parent().height():
            self.hide()
        else:
            self.move(int(self.x), int(self.y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(self.color)
        painter.drawEllipse(0, 0, self.size, self.size)

class FingerCountingApp(QWidget):
    def __init__(self, questions, switch_to_dashboard_callback):
        super().__init__()
        self.questions = random.sample(questions, 5)
        self.switch_to_dashboard = switch_to_dashboard_callback
        self.question_number = 0
        self.current_fingers = 0
        self.answer_shown = False
        
        # Load finger counting model
        self.model = load_model("dependencies\\model_30.keras")
        self.detector = HandDetector(maxHands=2)
        self.imgSize = 300
        self.offset = 20
        self.IMG_MODEL_SIZE = 128
        
        self.initUI()
        self.initCamera()

    def initUI(self):

        self.setGeometry(100, 100, 800, 650)
        qimg = QImage(underwater.data, underwater.shape[1], underwater.shape[0], 
                      underwater.shape[1]*3, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        
        self.background_label = QLabel(self)
        self.background_label.setPixmap(pixmap)
        self.background_label.setGeometry(0, 0, 800, 600)
        self.background_label.lower()

        self.camera_label = QLabel(self)
        self.camera_label.setGeometry(75, 120, 640, 400)
        self.camera_label.setStyleSheet("border: 3px solid #2e3b4e; border-radius: 15px; border-radius: 0px;")

        self.question_label = QLabel(self.questions[self.question_number][0], self)
        self.question_label.setGeometry(50, 20, 700, 60)
        self.question_label.setStyleSheet("font-size: 28px; font-family: Comic Sans MS; color: #f5e9bd; background-color: #2e3b4e; padding: 10px 10px 10px 10px; border-top-left-radius: 15px; border-top-right-radius: 15px;")
        add_shadow(self.question_label)

        self.answer_label = QLabel("Show your answer with your fingers!", self)
        self.answer_label.setGeometry(50, 80, 700, 30)
        self.answer_label.setStyleSheet("font-size: 18px; color: #f5e9bd; font-family: Arial; background-color: #2e3b4e; padding: 0px 10px 5px 10px; border-bottom-left-radius: 15px; border-bottom-right-radius: 15px;")
        add_shadow(self.answer_label)

        self.next_button = QPushButton("Next Question ➡️", self)
        self.next_button.setGeometry(525, 530, 200, 50)
        self.next_button.setStyleSheet("background-color: #a8e3b3; font-size: 18px; font-family: Comic Sans MS; border-radius: 10px; border-radius: 15px;")
        self.next_button.clicked.connect(self.next_question)
        add_shadow(self.next_button)

        self.back_button = QPushButton("⬅️ Go To Main Menu", self)
        self.back_button.setGeometry(65, 530, 200, 50)
        self.back_button.setStyleSheet("background-color: #87CEFA; font-size: 18px; font-family: Comic Sans MS; border-radius: 10px; border-radius: 15px;")
        self.back_button.clicked.connect(self.back_to_dashboard)
        add_shadow(self.back_button)

        self.reward_button = QPushButton("🍱 You Did Great!", self)
        self.reward_button.setGeometry(295, 530, 200, 50)
        self.reward_button.setStyleSheet("background-color: #c98ed8; font-size: 18px; font-family: Comic Sans MS; border-radius: 10px; border-radius: 15px;")
        self.reward_button.setVisible(False)
        self.reward_button.clicked.connect(self.reward)
        add_shadow(self.reward_button)

        self.speak_question(self.questions[self.question_number][0])
    
    def speak_question(self, question):
        self.speech_thread = SpeechThread(question)
        self.speech_thread.speak_signal.connect(self.on_speech_completed)
        self.speech_thread.start()

    def on_speech_completed(self):
        pass

    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        hands, _ = self.detector.findHands(frame)
        total_fingers = 0

        if hands:
            for hand in hands:
                x, y, w, h = hand['bbox']
                imgCrop = frame[y-self.offset:y+h+self.offset, x-self.offset:x+w+self.offset]
                
                imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        k = self.imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                        wGap = (self.imgSize - wCal) // 2
                        imgWhite[:, wGap:wCal+wGap] = imgResize
                    else:
                        k = self.imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                        hGap = (self.imgSize - hCal) // 2
                        imgWhite[hGap:hCal+hGap, :] = imgResize

                    imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
                    imgInput = cv2.resize(imgGray, (self.IMG_MODEL_SIZE, self.IMG_MODEL_SIZE))
                    imgInput = imgInput / 255.0
                    imgInput = np.expand_dims(imgInput, axis=(0, -1))

                    prediction = self.model.predict(imgInput)
                    total_fingers += np.argmax(prediction)

                except:
                    continue

        if total_fingers > 0:
            correct = self.questions[self.question_number][1]
            if total_fingers == correct and not self.answer_shown:
                self.answer_label.setText(f"✅ Correct! You showed {total_fingers} fingers!")
                self.answer_shown = True
            elif not self.answer_shown:
                self.answer_label.setText(f"❌ You showed {total_fingers} fingers. Try again!")

        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], 
                     frame.shape[1]*3, QImage.Format_BGR888)
        self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def count_fingers(self, hand_landmarks, hand_label):
        fingers = 0
        finger_tips = [8, 12, 16, 20]
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers += 1

        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]

        if hand_label == "Right":
            if thumb_tip.x < thumb_mcp.x:
                fingers += 1
        else:
            if thumb_tip.x > thumb_mcp.x:
                fingers += 1

        return fingers

    def next_question(self):
        if not self.answer_shown:
            return
        self.question_number += 1
        if self.question_number < len(self.questions):
            self.question_label.setText(self.questions[self.question_number][0])
            self.answer_label.setText("Show your answer with your fingers!")
            self.answer_shown = False
            self.speak_question(self.questions[self.question_number][0])
        else:
            self.question_label.setText("🎉 You’ve completed all the questions!")
            self.answer_label.setText("Click the reward button below! 🍱")
            self.reward_button.setVisible(True)

    def reward(self):
        self.answer_label.setText("⭐ Amazing job! You're a finger counting master!")
        animation = QPropertyAnimation(self.reward_button, b"geometry")
        animation.setDuration(1000)
        animation.setStartValue(self.reward_button.geometry())
        animation.setEndValue(QRect(self.reward_button.x() - 10, self.reward_button.y() - 10,
                                    self.reward_button.width() + 20, self.reward_button.height() + 20))
        animation.start()

        if not hasattr(self, 'confetti_timer'):
            self.confetti_particles = []
            self.confetti_timer = QTimer(self)
            self.confetti_timer.timeout.connect(self.update_confetti)
            self.confetti_timer.start(30)

            for _ in range(100):
                particle = ConfettiParticle(self)
                self.confetti_particles.append(particle)

    def update_confetti(self):
        for particle in self.confetti_particles:
            particle.move_down()

    def back_to_dashboard(self):
        self.cap.release()
        self.timer.stop()
        self.switch_to_dashboard()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finger Counting Game")
        self.setGeometry(100, 100, 800, 650)
        self.layout = QStackedLayout()
        self.setLayout(self.layout)

        self.dashboard = QWidget()

        # Convert the underwater image to QImage
        qimg = QImage(underwater.data, underwater.shape[1], underwater.shape[0], underwater.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Create a QLabel for the background image
        self.dashboard_label = QLabel(self)
        self.dashboard_label.setPixmap(pixmap)
        self.dashboard_label.setGeometry(0, 0, 800, 600)
        self.dashboard_label.lower()  # Ensure the background stays behind other widgets

        self.stage1_button = QPushButton("GENERAL\n KNOWLEDGE\n AND LANGUAGE", self.dashboard)
        self.stage1_button.setGeometry(75, 190, 300, 270)
        self.stage1_button.setStyleSheet("""background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #86a6dd, stop:1 #b9cae7); 
                                         font-size: 25px; font-family: Comic Sans MS; border-radius: 10px;""")
        self.stage1_button.clicked.connect(self.start_stage1)
        add_shadow(self.stage1_button)

        self.stage2_button = QPushButton("MATH AND\n FINGER\n COUNTING", self.dashboard)
        self.stage2_button.setGeometry(425, 190, 300, 270)
        self.stage2_button.setStyleSheet("""background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b9cae7, stop:1 #86a6dd); 
                                         font-size: 25px; font-family: Comic Sans MS; border-radius: 10px;""")
        self.stage2_button.clicked.connect(self.start_stage2)
        add_shadow(self.stage2_button)

        self.layout.addWidget(self.dashboard)

    def start_stage1(self):
        self.stage = FingerCountingApp(STAGE1_QUESTIONS, self.back_to_dashboard)
        self.layout.addWidget(self.stage)
        self.layout.setCurrentWidget(self.stage)

    def start_stage2(self):
        self.stage = FingerCountingApp(STAGE2_QUESTIONS, self.back_to_dashboard)
        self.layout.addWidget(self.stage)
        self.layout.setCurrentWidget(self.stage)

    def back_to_dashboard(self):
        self.layout.setCurrentWidget(self.dashboard)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()