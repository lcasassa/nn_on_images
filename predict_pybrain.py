classification = False
use_net = True
import cPickle as pickle

import cv2
import glob
import sys
if classification:
    from pybrain.datasets import ClassificationDataSet
else:
    from pybrain.datasets import SupervisedDataSet
import fitness
from random import shuffle
import os
import feature
import numpy as np
show_only_bad_images = False

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
#sorted(images_path)

malo = 0
bueno = 0
total = 0

do_all = False

images_path = sys.argv[1] if len(sys.argv) == 2 else "data2/**/*.jpg"
fitness.setData(images_path)

for input_data, output_data, image_path in fitness.getNextData(recalc=True, return_image_path=True, use_images_without_output=use_net):
    total += 1

    image = cv2.imread(image_path)

    if use_net:
        if classification:
            ds = ClassificationDataSet(len(input_data), nb_classes=2, class_labels=['aceptado', 'despunte'])
            ds.addSample(features, [0])
            ds._convertToOneOfMany()
            out = net.activateOnDataset(ds)
            out_class = out.argmax(axis=1)  # the highest output activation gives the class
        else:
            ds = SupervisedDataSet(len(input_data), net.indim)
            ds.addSample(input_data, [0]*net.indim)
            out = net.activateOnDataset(ds)[0]
            print out

    debug_image = []
    if output_data is not None:
        debug_image.extend(feature.debug_feature(output_data, image_path))

    if use_net:
        debug_image.extend(feature.debug_feature(out, image_path))

    for img in debug_image:
        # img = cv2.cvtColor(((img / img.max()) * 255.0).astype(np.float32), cv2.COLOR_GRAY2BGR)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        if not (len(img.shape) >= 3 and img.shape[2] == 3):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        image = concatenate(image, img, horizontal=True)

    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    cv2.imshow("Press key", image)

    key = cv2.waitKey(0 if not do_all else 1) % (256 * 2)
    if key == 27:
        break
    continue

    #out = out.reshape(X.shape)


    """
    text = "  ".join(["%.1f" % (x * 100.0) for x in out[0]])

    cv2.putText(image, text, (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
    red = str(ds.class_labels[out_class[0]])
    """
    red = "despunte" if features[0] > 0 else "aceptado"
    real = os.path.basename(os.path.dirname(image_path))

    text = ""
    text2 = "red:" + red
    text3 = "real:" + real

    print text3, text2, text  # , "=", sum([x * 100.0 for x in out[0]]), image_path

    if red != real:
        malo += 1
    else:
        bueno += 1

    if red != real or not show_only_bad_images:
        #print "Mal clasificado!"
        cv2.putText(image, text2, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        cv2.putText(image, text3, (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

        resize = 0.5
        down = 0.3
        shape = [x*resize for x in image.shape]
        shape[0] = float(shape[0])*down

        y_offset = int(float(image.shape[0])*(1-down))
        x_offset = 0

        #print down, 1-down, image.shape[0], y_offset

        cv2.rectangle(image, (x_offset, y_offset), (image.shape[1], image.shape[0]), (0, 255, 0), 3)
        #features = [image.shape[1]*resize*50-1]
        """
        for f in features:
            y = np.floor(f / shape[1])/resize
            x =(f % shape[1])/resize
            cv2.circle(image, (int(x) + x_offset, int(y) + y_offset), 1, (0,0,255), -1)
        """

        for img in debug_image:
            #img = cv2.cvtColor(((img / img.max()) * 255.0).astype(np.float32), cv2.COLOR_GRAY2BGR)
            img = img.astype(np.float32)
            img = (img-img.min())/(img.max()-img.min())*255.0
            img = img.astype(np.uint8)
            if not (len(img.shape) >= 3 and img.shape[2] == 3):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image = concatenate(image, img, horizontal=True)

        #image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

        cv2.imshow("Press key", image)

        key = cv2.waitKey(0 if not do_all else 1) % (256*2)
    else:
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
    #print key

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
