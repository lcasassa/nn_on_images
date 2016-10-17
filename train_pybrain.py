classification = False
import os
import cPickle as pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
#from pybrain.structure.modules import SoftmaxLayer
#from pybrain.utilities import percentError
if classification:
    from pybrain.datasets import ClassificationDataSet
else:
    from pybrain.datasets.supervised import SupervisedDataSet

continue_training = False
use_old_fitness = False
validation_percentage = 0.20


import fitness

fitness.setData("data2/**/*.jpg")
print "Getting data... Total images:", len(fitness.images_path)
ds = None
count = 0
for input_data, output_data in fitness.getNextData(recalc=not use_old_fitness):
    if ds is None:
        if classification:
            ds = ClassificationDataSet(len(input_data), nb_classes=len(fitness.outputsClass), class_labels=fitness.outputsClass)
        else:
            ds = SupervisedDataSet(len(input_data), len(output_data))

    ds.addSample(input_data, output_data)
    if count%10 == 0:
        print count,
    count += 1
print ""
if classification:
    ds._convertToOneOfMany()
    print "total count:", ds.calculateStatistics()

print "indim:", ds.indim, "outdim:", ds.outdim, "rows:", ds.endmarker['input']

output_model_file = 'model.pkl'

hidden_layers = [300, 100]
print "hidden_layers:", hidden_layers

#tstdata, trndata = ds.splitWithProportion(validation_percentage )
#trndata._convertToOneOfMany( )
#tstdata._convertToOneOfMany( )

#print "verify data rows:", tstdata.endmarker['input']
#print "train data rows:", trndata.endmarker['input']

if continue_training and os.path.isfile('oliv.xml'):
    print "Loading network"
    fnn = NetworkReader.readFrom('oliv.xml')
else:
    print "Creating new network"
    args = [ds.indim]
    args.extend(hidden_layers)
    args.append(ds.outdim)
    fnn = buildNetwork(*args)#, outclass=SoftmaxLayer, recurrent=False, bias=True)

trainer = BackpropTrainer( fnn, dataset=ds, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)

while True:
    try:
        print "Number of epochs o 's' to save or 'c' to save and exit or 't' to test on sample data or 'y' to test on "
        key = raw_input()
        if key == 's':
            NetworkWriter.writeToFile(fnn, 'oliv.xml')
            pickle.dump(fnn, open(output_model_file, 'wb'))
        elif key == 'c':
            break
        elif key == 'p':
            print
        elif key == 'x':
            import cv2

            WINDOW_NAME="Close window to continue training"

            for image_path in fitness.images_path:
                image = cv2.imread(image_path)
                import feature
                input_data = feature.calculate_feature(image_path)
                if input_data is None:
                    continue
                if classification:
                    ds = ClassificationDataSet(len(input_data), nb_classes=len(fitness.outputsClass), class_labels=fitness.outputsClass)
                else:
                    ds = SupervisedDataSet(len(input_data), len(feature.get_output(image_path)))
                ds.addSample(input_data, [0])
                output_data = fnn.activateOnDataset(ds)[0]

                text = "%.1f      %.1f" % (output_data[0] * 100.0, output_data[1] * 100.0)
                cv2.putText(image, text, (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(0) % 512
                print key
                if key == 511:
                    break

        else:
            epochs = int(key)
            trainer.trainUntilConvergence(dataset=ds, maxEpochs=epochs, verbose=True, continueEpochs=10,
                                  validationProportion=validation_percentage)

            #trainer.trainEpochs(epochs)
            #print 'Percent Error on Test dataset: ', percentError(trainer.testOnClassData(
            #    dataset=tstdata)
            #    , tstdata['class'])

        """
        elif key == 'y':
            output = fnn.activateOnDataset(trndata)
            for i in xrange(len(trndata['target'])):
                print ["%.3f" % x for x in trndata['target'][i]],
                print ["%.3f" % x for x in output[i]]
                #print ["%.3f" % x for x in trndata['target'][i] - output[i]]
            print "Total error:", percentError(trainer.testOnClassData(
                dataset=trndata)
                , trndata['class'])
        elif key == 't':
            output = fnn.activateOnDataset(tstdata)
            for i in xrange(len(tstdata['target'])):
                print ["%.3f" % x for x in tstdata['target'][i]],
                print ["%.3f" % x for x in output[i]]
                #print ["%.3f" % x for x in tstdata['target'][i] - output[i]]
            print "Total error:", percentError(trainer.testOnClassData(
                dataset=tstdata)
                , tstdata['class'])
        """
    except ValueError:
        print "Not a number"

NetworkWriter.writeToFile(fnn, 'oliv.xml')

pickle.dump( fnn, open( output_model_file, 'wb' ))
