import cv2
import numby as np
import pyttsx3
import threading



def talk_function(data):
	engine=pyttsx3.init()
	engine.say(data)
	engine.runAndWait()


faceModel=cv2.CascadeClassifier("ai_face_brain.xml")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()#RGB

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#x,y,w,hRGB==BGR convert gray
    faces=faceModel.detectMultiScale(gray,1.3,5)
    #slider incresed 1.3


    for(x,y,w,h) in faces:
    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    	cv2.putText(frame,'Face',(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    	#frame[x:x+w,y:y+h]=cv2.medianBlur(frame[x:x+w,y:y+h],35)
    	#frame[0:100,0:200]=cv2.medianBlur(frame[0:100,0:200],35)
    	 #frame,x,y,width,height,color,thickness


    	x = threading.Thread(target=talk_function, args=("hi,you are looking good")).start()
		

    # numberoffaces=len(faces)
    # if numberoffaces>=2:
    # 	playsound.playsound('English_song.mp3')


    cv2.imshow('Frame', frame)
    #cv2.imshow('grayFrame', gray)


    if cv2.waitKey(10)== ord('q'):
        break


cap.release()
cv2.destroyAllWindows()