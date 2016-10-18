import numpy as np
import os
import json
import cv2
import sys


# image path to numpy array for input to a net
def get_input(image, return_image=False):
    debug_image = []
    #image = cv2.imread(image_path)
    imagel = image[0:100,140:175]
    imager = image[0:100,845:880]
    imagel = cv2.cvtColor(imagel, cv2.COLOR_BGR2LAB)[:,:,0]
    imager = cv2.cvtColor(imager, cv2.COLOR_BGR2LAB)[:,:,0]

    debug_image.append(imagel)
    debug_image.append(imager)
    input_data = np.concatenate((imagel.flatten(), imager.flatten()))
    if return_image:
        return input_data, debug_image
    return input_data
    #output_data os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
    #return [outputsClass.index(output_data)]


def get_feature(image_path, values=[1,0,0,0,1,0], recalc=False, return_image=False):
    data_path = image_path.rsplit('.', 1)[0] + '.npy'
    if not recalc and os.path.isfile(data_path):
        return np.load(data_path)

    desp = calculate_feature(image_path, value=values, return_image=return_image)

    if not (desp is None or desp[0] is None):
        if return_image:
            np.save(data_path, desp[0])
        else:
            np.save(data_path, desp)
    return desp


def calculate_feature(image_path, value=[1,0,0,0,1,0], return_image=False):
    debug_image = []
    json_path = image_path.rsplit('.',1)[0] + '.json'
    if not os.path.isfile(json_path):
        features = None
        if return_image:
            return features, debug_image
        return features

    with open(json_path) as data_file:
        data = json.load(data_file)

    xl = []
    yl = []
    xr = []
    yr = []
    if len(data) == 48:
        for i in xrange(len(data) / 6):
            tmp = data[i * 6:i * 6 + 6]
            xl.append(tmp[0][0])
            yl.append(tmp[0][1])
            xr.append(tmp[5][0])
            yr.append(tmp[5][1])
    elif len(data) == 8:
        for i in xrange(len(data)):
            if i<4:
                xl.append(data[i][0])
                yl.append(data[i][1])
            else:
                xr.append(data[i][0])
                yr.append(data[i][1])
    elif len(data) == 16:
        if abs(data[0][0] - data[1][0]) < 100:
            for i in xrange(len(data)):
                if i < 8:
                    xl.append(data[i][0])
                    yl.append(data[i][1])
                else:
                    xr.append(data[i][0])
                    yr.append(data[i][1])
        else:
            for i in xrange(len(data)):
                if i % 2 == 0:
                    xl.append(data[i][0])
                    yl.append(data[i][1])
                else:
                    xr.append(data[i][0])
                    yr.append(data[i][1])

    from lmfit.models import LinearModel
    modl = LinearModel()
    parsl = modl.guess(xl, x=yl)
    outl = modl.fit(xl, parsl, x=yl)
    #print(outl.fit_report(min_correl=0.25))

    ml = outl.best_values['slope']
    bl = outl.best_values['intercept']


    modr = LinearModel()
    parsr = modr.guess(xr, x=yr)
    outr = modr.fit(xr, parsr, x=yr)
    #print(outr.fit_report(min_correl=0.25))

    mr = outr.best_values['slope']
    br = outr.best_values['intercept']

    #x = m*y + b
    def fl(y):
        return int(ml*y+bl)

    def fr(y):
        return int(mr*y+br)


    image = cv2.imread(image_path)
    p1 = (fl(0), 0)
    p2 = (fl(image.shape[0]), image.shape[0])
    p3 = (fr(0), 0)
    p4 = (fr(image.shape[0]), image.shape[0])

    features = np.array([p1[0], p2[0], p3[0], p4[0]])

    if return_image:
        debug_image.append(apply_feature(features, cv2.imread(image_path)))

    if return_image:
        return features, debug_image
    return features


def apply_feature(features, image):
    #debug_image = []

    #input_data = get_input(image_path)

    #print "size of input:", len(input_data)

    #image = cv2.imread(image_path)
    p1 = (int(features[0]), 0)
    p2 = (int(features[1]), image.shape[0])
    p3 = (int(features[2]), 0)
    p4 = (int(features[3]), image.shape[0])

    """
    cv2.line(image, p1, p2, (255,0,0), 1)
    cv2.line(image, p3, p4, (255,0,0), 1)

    cv2.circle(image, p1, 10, (0, 255, 255), -1)
    cv2.circle(image, p2, 10, (0, 255, 255), -1)
    cv2.circle(image, p3, 10, (0, 255, 255), -1)
    cv2.circle(image, p4, 10, (0, 255, 255), -1)
    """
    #for i in xrange(len(xl)):
    #    cv2.circle(image,(xl[i],yl[i]), 10, (0,0,255), -1)
    #    cv2.circle(image,(xr[i],yr[i]), 10, (0,0,255), -1)
    #debug_image.append(image)

    pts1 = np.float32([p1,p2,p3,p4])
    h = p4[1] - p1[1]
    w = p4[0] - p1[0]
    pts2 = np.float32([[0,0],[0,h],[w,0],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image, M, (700,image.shape[0]))
    #debug_image.append(dst)

    return dst


from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.modules import SoftmaxLayer
import cPickle as pickle
import time
net = None
model_file = 'model_cobre.pkl'


def load_net():
    global net
    if os.path.isfile(model_file):
        print "Loading net."
        time_start = time.time()
        net = pickle.load(open(model_file, 'rb'))
        print time.time() - time_start, "seconds to load net detector catodo"
    else:
        print "No net model for detector de catodo. File does not exist:", model_file


def predict_feature(input_data):
    global net
    if net is None:
        load_net()
        if net is None:
            return None
    ds = SupervisedDataSet(len(input_data), net.indim)
    ds.addSample(input_data, [0] * net.indim)
    out = net.activateOnDataset(ds)[0]
    return out


def train_net(data, validation_percentage=0.20, hidden_layers=[300, 100], epochs=100):
    #data = [[in, out],[in2, out2]]
    ds = None
    print "Loading data"
    for input_data, output_data in data:
        if ds is None:
            ds = SupervisedDataSet(len(input_data), len(output_data))
        ds.addSample(input_data, output_data)
        print ".",
        sys.stdout.flush()
    print ""

    print "Creating new network"
    args = [ds.indim]
    args.extend(hidden_layers)
    args.append(ds.outdim)
    fnn = buildNetwork(*args, recurrent=False, bias=True) #outclass=SoftmaxLayer,

    trainer = BackpropTrainer(fnn, dataset=ds, momentum=0.1, learningrate=0.01, verbose=True, weightdecay=0.01)


    trainer.trainUntilConvergence(dataset=ds, maxEpochs=epochs, verbose=True, continueEpochs=10,
                              validationProportion=validation_percentage)

    print "Saving net to", model_file
    pickle.dump( fnn, open( model_file, 'wb' ))
    print "Done."


def train(images, recalc=False): # images = [[cv2 image, image path],[cv2 image2, image2 path]]
    data = getTrainData(images, recalc=recalc)
    train_net(data)


def getTrainData(images, recalc=False):
    for image in images:
        input_data, output_data = getData(image, recalc=recalc)
        if output_data is None:
            continue
        yield (input_data, output_data)


def getData(image, recalc=False):
    input_data = get_input(image[0]) # cv2 image
    output_data = get_feature(image[1], recalc=recalc) # path

    return input_data, output_data

def get_catodo(image):
    input_catodo = get_input(image)
    features = predict_feature(input_catodo)
    image_catodo = apply_feature(features, image)
    return image_catodo

#pickle.dump(fnn, open(output_model_file, 'wb'))

#if __name__ == "__main__":
#    print get_feature("/home/linus/innovaxxion/NEAT/python/examples/images/data/nodulo/20160721014112_0_17_full_image.jpg")
