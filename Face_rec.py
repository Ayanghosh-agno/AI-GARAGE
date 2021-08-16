import cv2
import train_face
import time
import os
import datetime
import credentials
import face_recognition
import RPi.GPIO as GPIO
print(cv2.__version__)
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(12, GPIO.OUT)
GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'
video=cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

Encodings=[train_face.donEncode,train_face.nancyEncode]
Names=["Donald Trump","Nancy Pelosi"]

font=cv2.FONT_HERSHEY_SIMPLEX
while True:
        check,frame=video.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
             facePositions=face_recognition.face_locations(frame)
             allEncodings=face_recognition.face_encodings(frame,facePositions)
             frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
             for (top,right,bottom,left),face_encoding in zip(facePositions,allEncodings):
                      name="UNKNOWN"
                      matches=face_recognition.compare_faces(Encodings,face_encoding)
                      if True in matches:
                           first_match_index=matches.index(True)
                           name=Names[first_match_index]
                           GPIO.output(12, False)
                           picname=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
                           cv2.imwrite(picname+".jpg",frame)
                           credentials.multi_part_upload("agnobucket", picname+".jpg",picname+".jpg")
                           json_document={"link":credentials.COS_ENDPOINT+"/"+"agnobucket"+"/"+picname+".jpg"}
                           new_document = credentials.my_database.create_document(json_document)
                           time.sleep(5)
                           GPIO.output(12, True)     
                      cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                      cv2.putText(frame,name,(left,top-6),font,.75,(0,255,255),2)
             cv2.imshow("Frame", frame)
             cv2.waitKey(0)
        
cv2.destroyAllWindows()
