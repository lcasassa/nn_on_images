import neat
import cv2
import glob
import os
import json
import numpy as np
import feature


inputsAmount = 0
outputsAmount = 0
images_path = []
outputsClass = []


def crop(image, y):
    x1 = 0
    x2 = image.shape[1]
    y1 = y-3
    y2 = y+3
    crop_image = image[y1:y2, 0:image.shape[1]]
    crop_image_res = cv2.resize(crop_image, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
    crop_image_res = cv2.cvtColor(crop_image_res, cv2.COLOR_BGR2HSV)
    return crop_image_res[:,:,0]


def setData(images_path_):
    global inputsAmount, outputsAmount, images_path, outputsClass
    images_path = sorted(glob.glob(images_path_))
    outputsClass = [os.path.basename(classdir).replace(" ", "_") for classdir in sorted(glob.glob(os.path.dirname(images_path_)))]

    """
    image_path = images_path[0]
    image = cv2.imread(image_path)
    crop_image = crop(image, 10)
    
    inputsAmount = crop_image.size
    outputsAmount = 6
    """
    inputsAmount = feature.get_feature_size()
    outputsAmount = len(outputsClass)


def getNextData(recalc=False):
    for image_path in images_path:
        """
        json_path = image_path.rsplit('.')[0] + '.json'
        if not os.path.isfile(json_path):
            continue
        with open(json_path) as data_file:    
            data = json.load(data_file)
        for i in xrange(len(data)/6):
            image = cv2.imread(image_path)
            x = [float(p[0])/image.shape[1] for p in data[i*6:i*6+6]]
            y = [p[1] for p in data[i*6:i*6+6]]
            y = sum(y)/len(y)
            crop_image = crop(image, y)
            #cv2.imwrite("la.jpg", crop_image)
            image = np.array(crop_image).astype(float).flatten()/255.0
            #print "image shape:", image.shape, crop.shape
            yield (image, x)
        """
        input_data = feature.get_feature(image_path, recalc=recalc)
        output_data = os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
        yield (input_data, outputsClass.index(output_data))


def experiment( orgm, verbose = False ):
    #import ipdb; ipdb.set_trace()
    error = 0
    for input_data, output_data in getNextData():
        orgm.ann.setInputs(list(input_data))
        orgm.ann.spread()
        output_ann = orgm.ann.getOutputs()
        if verbose:
            print "output difference: ",
        error_max = 0
        for i in xrange(6):
            error_ = abs(output_ann[i] - output_data[i])
            if error_max < error_:
                error_max = error_
            if verbose:
                print "%.2f" % abs(output_ann[i] - output_data[i]),
        error += error_max
        if verbose:
            print ""
    error_MAX = 4
    fitness = (error_MAX - error/2)**2
    if verbose:
        print "fitness: ", fitness
    return fitness


if __name__ == "__main__":
    setData("data/**/*.jpg")
    for input_data, output_data in getNextData():
        print output_data, len(input_data), input_data