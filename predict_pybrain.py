import cPickle as pickle

import cv2
import glob
import sys
from pybrain.datasets import ClassificationDataSet
import feature
from random import shuffle
import os
import numpy as np


model_file = 'model.pkl'

net = pickle.load(open(model_file, 'rb'))

images_path = glob.glob(sys.argv[1])
shuffle(images_path)

malo = 0
bueno = 0
total = 0

i=0
do_all = False

while True:
    image_path = images_path[i]
    features = feature.get_feature(image_path, recalc=True, return_image=True)
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
        cv2.imshow("Press key", image)

        print text3, text2, text, "=", sum([x * 100.0 for x in out[0]]), image_path


        key = cv2.waitKey(0 if not do_all else 1) % (256*2)
    else:
        bueno += 1
        #print "Bien clasificado:", red, real
        key = 336
    total += 1
    #print key

    if key == ord('g'):
        do_all = not do_all

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
