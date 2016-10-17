import cv2
import numpy as np


def detector_borde_plastico(image):
    debug_image = []
    debug_image.append(image)

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    L = image_lab[:,:,0]
    a = image_lab[:,:,1]
    b = image_lab[:,:,2]
    """
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(L, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    kernel1 = cv2.getGaborKernel(ksize=(131, 131), sigma=5, theta=0, lambd=28, gamma=0.2, psi=0, ktype=cv2.CV_32F)
    debug_image.append(kernel1)

    destL = cv2.filter2D(L, cv2.CV_32F, kernel1)
    desta = cv2.filter2D(a, cv2.CV_32F, kernel1)
    """

    debug_image.append(L)
    #debug_image.append(closing)
    #debug_image.append(destL)
    debug_image.append(a)
    #debug_image.append(desta)
    debug_image.append(b)

    return image, debug_image