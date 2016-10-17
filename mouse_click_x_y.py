import cv2
import os
import sys
import glob
import json
from random import shuffle


images_path = sorted(glob.glob(sys.argv[1]))
shuffle(images_path)

mouse = []

def callback_mouse(event, x, y, flags, param):
    global mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse.append((x,y))
        print 'mouse size:', len(mouse), (x, y)

cv2.namedWindow('Press a key')
cv2.setMouseCallback('Press a key', callback_mouse)

for image_path in images_path:
    mouse = []
    json_path = image_path.rsplit('.', 1)[0] + '.json'

    image = cv2.imread(image_path)

    text = 'yes' if os.path.isfile(json_path) else 'no'
    cv2.putText(image, text, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    y_lines = [10, 100, 200, 300, 400, 500, 600, 700]
    for y in y_lines:
        cv2.line(image, (0,y), (image.shape[1],y), (0,0,255), 2)

    cv2.imshow('Press a key', image)
    key = cv2.waitKey(0) % 256
    if key == 27:
        break
    key = chr(key)
    print key
    if key == 's':
        print 'Saving to', json_path, 'mouse clicks:', mouse
        with open(json_path, 'w') as outfile:
            json.dump(mouse, outfile)
