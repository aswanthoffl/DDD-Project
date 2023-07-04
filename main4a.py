import cv2
from tensorflow.keras.models import load_model 
import numpy as np  
from pygame import mixer 
import os
import face_recognition
import datetime 
import csv 
import time

os.chdir("/home/user/Downloads/DDDDeep/EY_Dataset/dataset_new")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


model = load_model('Eye_dataset/models/model_eyes1.h5')
model2 = load_model('yawn_dataset/models/model_yawn1.h5')


mixer.init()
sound= mixer.Sound(r'/home/user/Downloads/DDDDeep/alarm.wav')
cap = cv2.VideoCapture(0)

eye_score = 0
eye_thicc=2
yawn_score = 0
yawn_thicc=2
eye_counter=0
yawn_counter=0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL




# Load images and labels for face recognition
known_face_encodings = []
known_face_labels = []


img_path_1 = "/home/user/Downloads/DDDDeep/Ajnas.jpg"
img_path_2 = "/home/user/Downloads/DDDDeep/Aswanth_A.jpeg"

img_1 = cv2.imread(img_path_1)
img_2 = cv2.imread(img_path_2)


label_1 = "Ajnas"
label_2 = "Aswanth A"


known_face_encodings.append(face_recognition.face_encodings(img_1)[0])
known_face_encodings.append(face_recognition.face_encodings(img_2)[0])

known_face_labels = [label_1, label_2]

# Create CSV file and headers
with open('data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Eye Status', 'Yawn Status', 'Time'])

while True:
        ret, frame = cap.read()
        height,width = frame.shape[0:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #COLOR_BGR2GRAY
        faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=3)
        eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
        
        cv2.rectangle(frame, (0,height-50),(200,height),(0,0,0),thickness=cv2.FILLED)


        
         # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)


        # Loop over the detected faces and draw a rectangle around them
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if the detected face matches with any of the known faces
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_labels[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 6), font, 0.5, (0, 0, 255), 2)
       
        

        for (x,y,w,h) in faces:
                cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0),thickness=3 )
                mouth= frame[y:y+h, x:x+w]
                mouth= cv2.resize(mouth,(80, 80))
                mouth= mouth/255
                mouth=mouth.reshape(80, 80,3)
                mouth= np.expand_dims(mouth,axis=0)
                # preprocessing is done now model prediction
                yawn_prediction = model2.predict(mouth)
                print([np.round(x*100, 2) for x in yawn_prediction])
                # print(prediction)
                if yawn_prediction[0][0]<yawn_prediction[0][1]:
                        yawn_score=yawn_score+1
                        print("Yawn")
                        yawn_status = "Yawn"
                        cv2.putText(frame,'Yawn',(10,height-380),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,255), thickness=1,lineType=cv2.LINE_AA)
                else:
                        yawn_score=yawn_score-1
                        print("No Yawn")
                        yawn_status = "No_Yawn"
                        cv2.putText(frame,'No Yawn',(10,height-380),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,0,255),thickness=1,lineType=cv2.LINE_AA)

        for (ex,ey,ew,eh) in eyes:  
                cv2.rectangle(frame,pt1=(ex,ey),pt2=(ex+ew,ey+eh), color=(255,0,0), thickness=3 )
                # preprocessing steps
                eye= frame[ey:ey+eh,ex:ex+ew]
                eye= cv2.resize(eye,(80,80))
                eye= eye/255
                eye= eye.reshape(80,80,3)
                eye= np.expand_dims(eye,axis=0)
                # preprocessing is done now model prediction
                eye_prediction = model.predict(eye)
                #print(eye_prediction)
                # if eyes are closed
                if eye_prediction[0][0]>0.8:
                        eye_score=eye_score+1
                        eye_status = "Closed"
                        cv2.putText(frame,'closed',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255 ),thickness=1,lineType=cv2.LINE_AA)
                # if eyes are open
                else:
                        eye_score=eye_score-1
                        eye_status = "Open"
                        cv2.putText(frame,'open',(10,height-20),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)  

        if(yawn_score<0):
                        yawn_score=0
        cv2.putText(frame,'Score:'+str(yawn_score),(150,height-380), font,1,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Counter_y:'+str(int(yawn_counter/5)),(400,height-380), font, 1,(0,0,255),1,cv2.LINE_AA)
        if(yawn_score>7):
                #person is feeling sleepy so we beep the alarm
                if(yawn_counter/7>=3):
                        yawn_counter=0
                try:
                        sound.play()
                        
                        yawn_counter+=1
                except: # isplaying = False
                        pass
                if(yawn_thicc<8):
                        yawn_thicc= yawn_thicc+2
                else:
                        yawn_thicc=yawn_thicc-2
                        if(yawn_thicc<2):
                                yawn_thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),yawn_thicc)

        if(eye_score<0):
                        eye_score=0
        cv2.putText(frame,'Score:'+str(eye_score),(100,height-20), font,1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Counter_e:'+str(int(eye_counter/9)),(400,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)
        if(eye_score>15):
                #person is feeling sleepy so we beep the alarm
                if(eye_counter/8>=3):
                        eye_counter=0
                try:
                        sound.play()
                        eye_counter+=1

                except: # isplaying = False
                        pass
                if(eye_thicc<15):
                        eye_thicc= eye_thicc+2
                else:
                        eye_thicc=eye_thicc-2
                        if(eye_thicc<2):
                                eye_thicc=2
                                
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),eye_thicc)               
        cv2.imshow('frame',frame)
        if cv2.waitKey(33) & 0xFF==ord('q'):
                break

         # Get current time
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save data in CSV file
        with open('data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name,eye_status, yawn_status, current_time])           

cap.release()
cv2.destroyAllWindows()