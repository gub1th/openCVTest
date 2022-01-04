import numpy as np
import cv2
import dlib

def getMidpoint(p1, p2):
    return ((p1.x + p2.x)//2, (p1.y + p2.y)//2)

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/shapePredictorFaceLandmarks68.dat')

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        #x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        landmarks = predictor(gray, face)
        lEyeLeft = (landmarks.part(36).x, landmarks.part(36).y) #idk if left or right eye lol
        lEyeRight = (landmarks.part(39).x, landmarks.part(39).y)
        lEyeCenterTop = getMidpoint(landmarks.part(37), landmarks.part(38))
        lEyeCenterBot = getMidpoint(landmarks.part(41), landmarks.part(40))

        lEyeCenterTopX, lEyeCenterTopY = lEyeCenterTop
        lEyeCenterBotX, lEyeCenterBotY = lEyeCenterBot

        centerEyeX = landmarks.part(36).x + (landmarks.part(39).x - landmarks.part(36).x)//2
        centerEyeY = lEyeCenterTopY + (lEyeCenterBotY - lEyeCenterTopY)//2

        lEyeHorizontal = cv2.line(frame, lEyeLeft, lEyeRight, (255, 0, 0), 2)
        lEyeVertical = cv2.line(frame, lEyeCenterTop, lEyeCenterBot, (255, 0, 0), 2)

        lEyeCenter = cv2.circle(frame, (centerEyeX, centerEyeY), 5, (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break   

cap.release()
cv2.destroyAllWindows()
