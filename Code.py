import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img1 = cv2.imread("..\\img61.jpg")
img2 = cv2.imread("..\\Sample_Image.jpg")

# Convert to HSV
hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Define lower and uppper limits of what we call "white-ish"
sensitivity = 19
lower_white = np.array([0, 0, 255 - sensitivity])
upper_white = np.array([255, sensitivity, 255])

# Create mask to only select white
mask = cv2.inRange(hsv, lower_white, upper_white)

# Change image to black where we found white
image2 = img2.copy()
image2[mask > 0] = (0, 0, 0)

#mask is a 2D numpy array thus reshaping it into 3d numpy array to perform bitwise operations ahead
mask = mask.reshape((mask.shape[0], mask.shape[1], 1))

mask_inv = cv2.bitwise_not(mask)
mask_inv = mask_inv.reshape((mask_inv.shape[0], mask_inv.shape[1], 1))

img_foreground=img2 & mask_inv
img_background=img1 & mask
res_img = img_background | img_foreground

mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
mask_inv=cv2.cvtColor(mask_inv,cv2.COLOR_BGR2RGB)
img_background=cv2.cvtColor(img_background,cv2.COLOR_BGR2RGB)
img_foreground=cv2.cvtColor(img_foreground,cv2.COLOR_BGR2RGB)
res_img=cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(16,12))
plt.subplot(2,2,1)
plt.imshow(mask)
plt.title('Mask Image')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
plt.imshow(mask_inv)
plt.title('Inverse Mask Image')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,3)
plt.imshow(img_background)
plt.title('Image background')
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,4)
plt.imshow(img_foreground)
plt.title('Image foreground')
plt.xticks([])
plt.yticks([])
plt.figure(figsize=(10,8))
plt.subplot(1,1,1)
plt.imshow(res_img)
plt.title('Composite Image')
plt.xticks([])
plt.yticks([])
