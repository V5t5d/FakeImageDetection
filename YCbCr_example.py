import cv2
import matplotlib.pyplot as plt

img = cv2.imread("image_a.jpg")

ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

Y, Cb, Cr = cv2.split(ycbcr)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1), plt.imshow(Y, cmap='gray'), plt.title("Y (Luminance)")
plt.subplot(1, 3, 2), plt.imshow(Cb, cmap='gray'), plt.title("Cb (Chrominance-Blue)")
plt.subplot(1, 3, 3), plt.imshow(Cr, cmap='gray'), plt.title("Cr (Chrominance-Red)")
plt.show()
