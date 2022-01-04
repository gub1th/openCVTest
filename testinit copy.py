import numpy as np
import cv2
import dlib
import math
import os

def nothing():
	pass 

def getMidpoint(p1, p2):
    return ((p1.x + p2.x)//2, (p1.y + p2.y)//2)

def distance(x1, y1, x2, y2):
    return int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

def expandEllipse(ptop, pbot, pleft, pright):
    a = distance(pleft.x, pleft.y, pright.x, pright.y)
    b = distance(pbot.x, pbot.y, ptop.x, ptop.y)

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# grab resolution dimensions and set video capture to it.
def get_dims(cap, stddimslist, res='1080p'):
    width, height = stddimslist["480p"]
    if res in stddimslist:
        width,height = stddimslist[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


def get_video_type(filename, videotypelist):
    filename, ext = os.path.splitext(filename)
    if ext in videotypelist:
      return  videotypelist[ext]
    return videotypelist['avi']

###

filename = 'video.avi'
frames_per_second = 24.0
res = '720p'
recordingActive = False

# Standard Video Dimensions Sizes
stdDims =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
videoType = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

###

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(filename, get_video_type(filename, videoType), 25, get_dims(cap, stdDims, res))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/shapePredictorFaceLandmarks68.dat')

lEyeList = [i for i in range(36, 42)] #36-41
rEyeList = [i for i in range(42, 48)] #42-47

while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  
    elif key == ord('r'):
        recordingActive = bool(-recordingActive)

    if recordingActive:
        out.write(frame)

    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    cv2.putText(frame, f'faces: {len(faces)}', (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.line(frame, (0, height//2), (width, height//2), (255, 0, 0), 2) #draw horizontal
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 0, 0), 2) #draw vertical

    for face in faces:
        #x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        landmarks = predictor(gray, face)

        #leftEye
        lEyeCenterTop = getMidpoint(landmarks.part(37), landmarks.part(38))
        lEyeCenterBot = getMidpoint(landmarks.part(41), landmarks.part(40))
        lEyeCenterLeft = landmarks.part(36)
        lEyeCenterRight = landmarks.part(39)

        lEyeCenterTopX, lEyeCenterTopY = lEyeCenterTop
        lEyeCenterBotX, lEyeCenterBotY = lEyeCenterBot

        lEyeCenterX = landmarks.part(36).x + (landmarks.part(39).x - landmarks.part(36).x)//2
        lEyeCenterY = lEyeCenterTopY + (lEyeCenterBotY - lEyeCenterTopY)//2 
        
        #rightEye
        rEyeCenterTop = getMidpoint(landmarks.part(43), landmarks.part(44))
        rEyeCenterBot = getMidpoint(landmarks.part(47), landmarks.part(46))
        rEyeCenterLeft = landmarks.part(42)
        rEyeCenterRight = landmarks.part(45)

        rEyeCenterTopX, rEyeCenterTopY = rEyeCenterTop
        rEyeCenterBotX, rEyeCenterBotY = rEyeCenterBot

        rEyeCenterX = landmarks.part(42).x + (landmarks.part(45).x - landmarks.part(42).x)//2
        rEyeCenterY = rEyeCenterTopY + (rEyeCenterBotY - rEyeCenterTopY)//2 

        #smaller (many points)
        lEye = [(landmarks.part(ele).x, landmarks.part(ele).y) for ele in lEyeList]
        rEye = [(landmarks.part(ele).x, landmarks.part(ele).y) for ele in rEyeList]
        
        #for bigger (with ellipse)
        lDistX = distance(lEyeCenterLeft.x, lEyeCenterLeft.y, lEyeCenterRight.x, lEyeCenterRight.y)
        lDistY = distance(lEyeCenterBotX, lEyeCenterBotY, lEyeCenterTopX, lEyeCenterTopY)

        rDistX = distance(lEyeCenterLeft.x, lEyeCenterLeft.y, lEyeCenterRight.x, lEyeCenterRight.y)
        rDistY = distance(rEyeCenterBotX, rEyeCenterBotY, rEyeCenterTopX, rEyeCenterTopY)
        
        #drawing bigger
        #cv2.ellipse(frame, (lEyeCenterX, lEyeCenterY), (lDistX, lDistY), 0, 0, 360, (0, 0, 255), 2)
        #cv2.ellipse(frame, (rEyeCenterX, rEyeCenterY), (rDistX, rDistY), 0, 0, 360, (0, 0, 255), 2)
        
        mask = np.zeros((height,width), np.uint8)

        leftEllipse = cv2.ellipse(mask, (lEyeCenterX, lEyeCenterY), (lDistX, lDistY), 0, 0, 360, (0, 0, 255), -1)
        rightEllipse = cv2.ellipse(mask, (rEyeCenterX, rEyeCenterY), (rDistX, rDistY), 0, 0, 360, (0, 0, 255), -1)

        # print(mask, "mask")
        # masked_data = cv2.bitwise_and(frame, frame, mask=leftEllipse)

        # _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

        # contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # print(contours, "contours")
        # x,y,w,h = cv2.boundingRect(contours[0])
        # crop = masked_data[y:y+h,x:x+w]

        #drawing smaller
        for i in range(len(lEye)):
            cv2.line(frame, lEye[i], lEye[(i+1)%len(lEye)], (0, 255, 0), 2) #lEye
            cv2.line(frame, rEye[i], rEye[(i+1)%len(rEye)], (0, 255, 0), 2) #rEye
            
        
        

    #cv2.imshow('crop', crop)
    cv2.imshow('frame', frame) 

cap.release()
cv2.destroyAllWindows()
