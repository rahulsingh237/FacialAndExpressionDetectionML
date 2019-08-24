import cv2
import numpy as np
data=[]
capture = cv2.VideoCapture(0)
cascade=cv2.CascadeClassifier('C:\\Users\\rahul\\Desktop\\haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_COMPLEX
while True:
	ret,image=capture.read()
	if ret:
		#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces=cascade.detectMultiScale(image,1.3)
		for x,y,w,h in faces:
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),5)
			myFace=image[y:y+h,x:x+w,:]
			myFace =cv2.resize(myFace,(50,50))
			if len(data)<100:
				print(len(data))
				data.append(myFace)
		cv2.imshow('image.jpg',image)
		if cv2.waitKey(1) & 0xff==27 or len(data)>=100:
			break
arr=np.array(data)
print(arr.shape)
np.save('a.npy',arr)
capture.release()
face=data[0]
import matplotlib.pyplot as plot
plot.imshow(face)
plot.show()
cv2.waitKey(100)
cv2.destroyAllWindows()