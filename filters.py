import cv2
import numpy as np
import matplotlib.pyplot as plt
###loading image 
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