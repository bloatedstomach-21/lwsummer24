import subprocess
import pyttsx3
import smtplib
import cv2
imagePath = 'input_image.jpg'

from googlesearch import search
#1_________________________________________________________
def sendEmail():
    email_id = input("Enter your email id:")
    password = input("enter your email app password")
    server =smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
	
    server.login(email_id, password)
	
    to = input("Enter receipient email id: ")

    content = input("Enter body: ")
    server.sendmail(email_id, to, content)
    server.close()
    return True
#2_________________________________________________________
def searchresults1(searchtext):
	search(searchtext,num_results=5)
#3_________________________________________________________
def textmessages():
	client = vonage.Client(key="^^^^^$#@", secret="*****")
	sms = vonage.Sms(client)
	responseData = sms.send_message(
    {
        "from": "Vonage APIs",
        "to": "918234456333",
        "text": "A text message sent using the Nexmo SMS API",
    }
)

if responseData["messages"][0]["status"] == "0":
    print("Message sent successfully.")
else:
    print(f"Message failed with error: {responseData['messages'][0]['error-text']}")










#6_________________________________________________________
volume=input("enter the amount of volume in words or in %")
pwshellcmd="Setvol"+volume
subprocess.run(pwshellcmd)

#2_________________________________________________________
def textToSpeech(inputstring):
    engine=pyttsx3.init()
    engine.say(inputstring)
    engine.runAndWait()
#3__________________________________________________________

#12_________________________________________________________
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
cap = cv2.VideoCapture(0) 
while 1: 
	ret, img = cap.read() 

	# convert to gray scale of each frames 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

	# Detects faces of different sizes in the input image 
	faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

	for (x,y,w,h) in faces: 
		# To draw a rectangle in a face 
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
		roi_gray = gray[y:y+h, x:x+w] 
		roi_color = img[y:y+h, x:x+w] 

		# Detects eyes of different sizes in the input image 
		eyes = eye_cascade.detectMultiScale(roi_gray) 

		#To draw a rectangle in eyes 
		for (ex,ey,ew,eh) in eyes: 
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 

	# Display an image in a window 
	cv2.imshow('img',img) 

	# Wait for Esc key to stop 
	k = cv2.waitKey(30) & 0xff
	if k == 27: 
		break

# Close the window 
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 
#14____________________________________________-
sunglass_image = cv2.imread("path/to/sunglass_image.png", cv2.IMREAD_UNCHANGED)
	destination_image = cv2.imread("path/to/destination_image.jpg")

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray_image = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected in the destination image.")
        return

    x, y, w, h = faces[0]

    # Resize the sunglass image to match the face's width and adjust the position
    sunglass_resized = cv2.resize(sunglass_image, (w, int(0.4 * h)))
    x_offset = x
    y_offset = y + int(0.3 * h)

    alpha_s = sunglass_resized[:, :, 3] / 255.0
    alpha_d = 1.0 - alpha_s
    for c in range(0, 3):
        destination_image[y_offset:y_offset + sunglass_resized.shape[0], x_offset:x_offset + sunglass_resized.shape[1], c] = (
                    alpha_s * sunglass_resized[:, :, c] + alpha_d * destination_image[y_offset:y_offset + sunglass_resized.shape[0], x_offset:x_offset + sunglass_resized.shape[1], c]
                )

    cv2.imwrite('output_image.jpg', destination_image)

    print("Sunglass attached! Check 'output_image.jpg'.")