import cv2
videosource=input("enter 0 for device camera")
match videosource:
    case "0":
        link=0
    case "1":
        link=1
    case _:
        link=input("enter the ip address along with the port and directory for the vidsource")
a2=cv2.VideoCapture(link)
while True:
    status, photo = a2.read()
    cv2.imshow("hello", photo)
    cv2.waitKey(50)