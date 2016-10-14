import cPickle as pickle

import cv2
import glob
import sys
from pybrain.datasets import ClassificationDataSet
import feature
from random import shuffle
import os
import numpy as np


def concatenate(image1, image2, horizontal=False):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    size = image1.shape[2] if len(image1.shape) > 2 else 1

    if horizontal:
        if size == 1:
            vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
        else:
            vis = np.zeros((max(h1, h2), w1 + w2, size), np.uint8)
        vis[:h1, :w1] = image1
        vis[:h2, w1:w1 + w2] = image2
    else:
        if size == 1:
            vis = np.zeros((h1 + h2, max(w1, w2)), np.uint8)
        else:
            vis = np.zeros((h1 + h2, max(w1, w2), size), np.uint8)
        vis[:h1, :w1] = image1
        vis[h1:h1 + h2, :w2] = image2
    return vis


model_file = 'model.pkl'

net = pickle.load(open(model_file, 'rb'))

images_path = glob.glob(sys.argv[1])
shuffle(images_path)

malo = 0
bueno = 0
total = 0

i=0
do_all = False

cv2.namedWindow('values')
def nothing(x):
    pass
cv2.createTrackbar('values0', 'values', 1000, 1000, nothing)
cv2.createTrackbar('values1', 'values', 0, 1000, nothing)
#cv2.createTrackbar('values2', 'values', 0, 1000, nothing)
cv2.createTrackbar('values3', 'values', 0, 1000, nothing)
cv2.createTrackbar('values4', 'values', 1000, 1000, nothing)
#cv2.createTrackbar('values5', 'values', 0, 1000, nothing)

while True:
    image_path = images_path[i]

    values = []
    values.append(float(cv2.getTrackbarPos('values0', 'values'))/1000.0)
    values.append(float(cv2.getTrackbarPos('values1', 'values'))/1000.0)
    #values.append(float(cv2.getTrackbarPos('values2', 'values'))/1000.0)
    values.append(1-(values[0]+values[1]))
    values.append(float(cv2.getTrackbarPos('values3', 'values'))/1000.0)
    values.append(float(cv2.getTrackbarPos('values4', 'values'))/1000.0)
    values.append(1-(values[3]+values[4]))
    #values.append(float(cv2.getTrackbarPos('values5', 'values'))/1000.0)
    print values

    features, debug_image = feature.get_feature(image_path, values=values, recalc=True, return_image=True)
    if features is None:
        continue

    ds = ClassificationDataSet(len(features), nb_classes=2, class_labels=['aceptado', 'despunte'])
    ds.addSample(features, [0])

    ds._convertToOneOfMany()
    out = net.activateOnDataset(ds)

    out_class = out.argmax(axis=1)  # the highest output activation gives the class
    #out = out.reshape(X.shape)
    image = cv2.imread(image_path)


    text = "  ".join(["%.1f" % (x * 100.0) for x in out[0]])

    cv2.putText(image, text, (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

    real = os.path.basename(os.path.dirname(image_path))
    red = str(ds.class_labels[out_class[0]])
    if red != real:
        malo += 1
        #print "Mal clasificado!"
        text2 = "red:" + red
        cv2.putText(image, text2, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        text3 = "real:" + real
        cv2.putText(image, text3, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        resize = 0.5
        down = 0.3
        shape = [x*resize for x in image.shape]
        shape[0] = float(shape[0])*down

        y_offset = int(float(image.shape[0])*(1-down))
        x_offset = 0

        print down, 1-down, image.shape[0], y_offset

        cv2.rectangle(image, (x_offset, y_offset), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        #features = [image.shape[1]*resize*50-1]
        for f in features:
            y = np.floor(f / shape[1])/resize
            x =(f % shape[1])/resize
            cv2.circle(image, (int(x) + x_offset, int(y) + y_offset), 1, (0,0,255), -1)

        for img in debug_image:
            #img = cv2.cvtColor(((img / img.max()) * 255.0).astype(np.float32), cv2.COLOR_GRAY2BGR)
            img = img.astype(np.float32)
            img = (img-img.min())/(img.max()-img.min())*255.0
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image = concatenate(image, img, horizontal=False)
        cv2.imshow("Press key", image)

        print text3, text2, text, "=", sum([x * 100.0 for x in out[0]]), image_path

        key = cv2.waitKey(0 if not do_all else 1) % (256*2)
    else:
        bueno += 1
        #print "Bien clasificado:", red, real
        key = 336
    total += 1

    while key == 489:
        key = cv2.waitKey(0) % (256*2)

    if key == ord('g'):
        do_all = not do_all
    elif key == ord('s'):
        import shutil
        if not os.path.exists(os.path.join('copy', os.path.dirname(image_path))):
            os.makedirs(os.path.join('copy', os.path.dirname(image_path)))

        shutil.copyfile(image_path, os.path.join('copy', image_path))
    elif key == 32:
        continue
    print key

    if key == 337:
        i -= 1
        if i <= 0:
            i = 0
    elif key == 27 or (key == 511 and not do_all):
        break
    else:
        i += 1
        if not i < len(images_path):
            i -= 1
            break

print bueno, "buenos de", total, "%.3f" % (float(bueno)/total*100)
print malo, "malos de", total, "%.3f" % (float(malo)/total*100)
