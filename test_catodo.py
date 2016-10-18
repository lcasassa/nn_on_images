import os
import detector_catodo
import cv2
import glob
import sys

if len(sys.argv) == 2:
    images_path = sys.argv[1]
else:
    images_path = "data2/**/*.jpg"

def get_images(images_path):
    images_path = glob.glob(images_path)
    for image_path in images_path:
        image = cv2.imread(image_path)
        yield (image, image_path)


detector_catodo.load_net()
for image, image_path in get_images(images_path):
    image_catodo = detector_catodo.get_catodo(image)

    cv2.imshow("catodo", image_catodo)
    key = cv2.waitKey(0)
    if key == 27:
        break



