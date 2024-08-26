import subprocess
import pyttsx3
import smtplib
import cv2
import geocoder
import matplotlib.pyplot as plt
from googlesearch import search

imagePath = 'input_image.jpg'


#_________________________________________________________
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
#_________________________________________________________
searchtext=input("enter the search text")
def searchresults1(searchtext):
    print(list(search(searchtext,num_results=5)))   
searchresults1(searchtext)
#_________________________________________________________
def textmsgusingphone(number,message):
    subprocess.run(f'adb shell am start -a android.intent.action.SENDTO -d sms:{number} --es sms_body "{message}"')
#_________________________________________________________
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

#_________________________________________________________
def send_bulk_emails(subject, body, to_emails):

    email_id = input("Enter your email id:")
    password = input("enter your email app password")
    server =smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    
    server.login(email_id, password)
    
    to_emails = input("Enter receipient email ids for bulk messaging: ")

    content = input("Enter body: ")
    for email in to_emails:
        server.sendmail(email_id, email, content)


#_________________________________________________________
def setvolume():

    volume=input("enter the amount of volume in words or in %, please ensure that the SetVol utility is installed and it's environment path is set")
    pwshellcmd="Setvol"+volume
    subprocess.run(pwshellcmd)

#_________________________________________________________
def textToSpeech(inputstring):
    engine=pyttsx3.init()
    engine.say(inputstring)
    engine.runAndWait()
#__________________________________________________________
def geocodee():
    g = geocoder.ip('me')
    print(g.latlng)

#_________________________________________________________
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0) 
while 1: 
    ret, img = cap.read() 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
 
    cv2.imshow('img',img) 
    k = cv2.waitKey(30) & 0xff #esc key seq
    if k == 27: 
        break
cv2.imwrite("faces.jpg",img[y:y+h,x:x+w])
cap.release() 

cv2.destroyAllWindows() 
#____________________________________________-
sunglass_image = cv2.imread("path/to/sunglass_image.png", cv2.IMREAD_UNCHANGED)
	destination_image = cv2.imread("path/to/destination_image.jpg")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray_image = cv2.cvtColor(destination_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("No face detected in the destination image.")
        return

    x, y, w, h = faces[0]
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
#________________________________________________________
def create_custom_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[25:75, 25:75] = [2, 255, 2]  # green square
    plt.imshow(image)
    plt.axis('off')
    plt.show()
#________________________________________________________
def filters():
    loaded_img = cv2.imread("WIN_20240825_19_09_21_Pro.jpg")
    loaded_img = cv2.cvtColor(loaded_img,cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(loaded_img,cmap="gray")
    plt.axis("off")
    plt.show()
#--------------------------
    Sepia_Kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])
    Sepia_Effect_Img = cv2.filter2D(src=loaded_img, kernel=Sepia_Kernel, ddepth=-1)
    plt.figure(figsize=(8,8))
    plt.imshow(Sepia_Effect_Img,cmap="gray")
    plt.axis("off")
    plt.show()
#---------------------------
    Sharpen_Kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    Sharpen_Effect_Img = cv2.filter2D(src=loaded_img, kernel=Sharpen_Kernel, ddepth=-1)
    plt.figure(figsize=(8,8))
    plt.imshow(Sharpen_Effect_Img,cmap="gray")
    plt.axis("off")
    plt.show()
