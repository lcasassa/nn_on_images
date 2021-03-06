import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from PIL import ImageEnhance
import scipy.ndimage
from scipy import signal
import mdp
import os
import json
import cv2
import detector_catodo

MAXDIM = 500


def getMaxActivated(im, ndim):
    dimx, dimy = np.shape(im)

    data = np.reshape(im, dimx * dimy)
    outdata = []
    ndatos = 0

    while (ndatos < MAXDIM):
        mindex = np.argmax(data)
        # print mindex
        ycord = mindex % dimy
        if (ycord > 50):
            outdata.append(mindex)
            ndatos = ndatos + 1
        data[mindex] = 0

    return outdata


def myEqualize(im):
    contr = ImageEnhance.Contrast(im)
    im = contr.enhance(1.5)
    bright = ImageEnhance.Brightness(im)
    im = bright.enhance(3)
    # im.show()
    return im


def createGabor(ksize, theta, spatFreq, sigma, phi):
    MEAN = 127
    A = 127

    outGabor = 1.0 * np.ones((ksize, ksize))
    th = theta * np.pi / 180

    summ = 0
    for i in range(ksize):
        for j in range(ksize):
            xa = i * np.cos(th) - j * np.sin(th)
            ya = i * np.sin(th) + j * np.cos(th)
            A = 127 * np.exp(-0.5 * (i - ksize / 2) ** 2 / (sigma ** 2)) * np.exp(
                -0.5 * (j - ksize / 2) ** 2 / (sigma ** 2))
            val = 1.0 * A * np.sin(2 * np.pi * spatFreq * xa + phi) + MEAN

            outGabor[i, j] = np.round(val)
            summ = summ + outGabor[i, j] ** 2

    return outGabor / np.sqrt(summ)


def rgb2hsv(image):
    return image.convert('HSV')


def imConvolve(imgc, kernel):
    dimx, dimy = np.shape(imgc)
    ksize, ksize2 = np.shape(kernel)

    conv = np.zeros(np.shape(imgc))
    for i in range(dimx):
        for j in range(dimy):
            ll = i - ksize / 2
            rl = i + ksize / 2 + 1
            ul = j - ksize / 2
            bl = j + ksize / 2 + 1
            if ((ll > 0) & (rl < dimx) & (ul > 0) & (bl < dimy)):
                conv[i, j] = np.sum(np.sum(np.matrix.dot(imgc[ll:rl, ul:bl], kernel)))
    return conv


# ---------------------------------------------------
# ---------------------------------------------------
import detector_cobre

def get_feature_size():
    return MAXDIM

def get_input(image_path, return_image=False):
    if False:
        return detector_catodo.get_input(image_path, return_image)
    else:
        image = cv2.imread(image_path)
        feature = detector_catodo.get_feature(image_path)
        image_catodo = detector_catodo.apply_feature(feature, image)
        return_values = detector_cobre.get_input(image_catodo, return_image=return_image)
        return return_values

    #output_data os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
    #return [outputsClass.index(output_data)]

def get_feature(image_path, values=[1,0,0,0,1,0], recalc=False, return_image=False):
    return detector_catodo.get_feature(image_path, values, recalc, return_image)


def calculate_feature(image_path, value=[1,0,0,0,1,0], return_image=False):
    return detector_catodo.calculate_feature(image_path, value, return_image)


def debug_feature(features, image_path):
    return detector_catodo.debug_feature(features, image_path)

def calculate_feature2(image_path, value=[1,0,0,0,1,0], return_image=False):
    debug_image = []
    image = cv2.imread(image_path)

    from detector_borde_plastico import detector_borde_plastico
    image_bp, images_dp_debug = detector_borde_plastico(image)
    debug_image.extend(images_dp_debug)

    image_bottom = image_bp[int(0.7*image_bp.shape[0]):image_bp.shape[0],:,:]

    #image_sharr = cv2.Scharr(image_bottom, cv2.CV_32F, 2, 0)

    image_bottom_lab = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2LAB)
    #features = image_bottom_lab.flatten()[:500]
    #debug_image.append(image_bottom_lab[:,:,0])
    #debug_image.append(image_bottom_lab[:,:,2])
    image_lab_a = image_bottom_lab[:, :, 1]
    image_lab_b = image_bottom_lab[:, :, 2]
    debug_image.append(image_lab_a)
    debug_image.append(image_lab_b)

    thresh = cv2.threshold(image_lab_a, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    #debug_image.append(thresh)

    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bigest_area = amg
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area:
            max_area = area
            biggest_area = cnt
    """
    # points = [(100,h1[1][1]-h1[0][1]-5), (200, h1[1][1]-h1[0][1]-5), (300, h1[1][1]-h1[0][1]-5)]
    bad = 0
    good = 0
    for x in xrange(0, th2.shape[1], 10):
        p = (x, th2.shape[0] - 5)
        if cv2.pointPolygonTest(biggest_area, p, False) <= 0:
            bad += 1
        else:
            good += 1

    # print "good:", good, "bad:", bad
    isgood = False
    if bad < 25:
        isgood = True
    """
    mask_pasto = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask_pasto, [biggest_area], 0, 255, 3)
    debug_image.append(mask_pasto)

    #features = [mask_pasto == 1]

    """
    image_kmeans = kmeans(image_bottom_lab, K=40)
    image_kmeans_rgb = cv2.cvtColor(image_kmeans, cv2.COLOR_LAB2BGR)

    debug_image.append(image_kmeans_rgb)

    image_kmeans_rgb_cut = image_kmeans_rgb[0:image_kmeans_rgb.shape[0], int(image_kmeans_rgb.shape[1]*0.167):int(image_kmeans_rgb.shape[1]*0.905)]
    debug_image.append(image_kmeans_rgb_cut)

    image_kmeans2 = kmeans(image_kmeans_rgb_cut, K=5)
    #image_kmeans2_rgb = cv2.cvtColor(image_kmeans2, cv2.COLOR_LAB2BGR)
    debug_image.append(image_kmeans2)



    lowerBound = np.array(value[0:3])*255.0
    upperBound = np.array(value[3:6])*255.0

    # this gives you the mask for those in the ranges you specified,
    # but you want the inverse, so we'll add bitwise_not...
    image_inrange = cv2.inRange(image_kmeans2, lowerBound, upperBound)
    image_inrange_color = cv2.bitwise_not(image_kmeans_rgb_cut, mask=image_inrange)
    debug_image.append(image_inrange_color)

    """


    """
    cv2.kmeans(image_bottom_lab, 40, )
    from sklearn.cluster import MiniBatchKMeans
    image_bottom_lab_reshape = image_bottom_lab.reshape((image_bottom_lab.shape[0] * image_bottom_lab.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=10)
    labels = clt.fit_predict(image_bottom_lab_reshape)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape(image_bottom_lab.shape)
    image_bottom_lab_res = image_bottom_lab_reshape.reshape(image_bottom_lab.shape)

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image_bottom_lab_res = cv2.cvtColor(image_bottom_lab_res, cv2.COLOR_LAB2BGR)


    sobelx64f = cv2.Sobel(image_bottom, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    lowerBound = np.array(value[0:3])*255.0
    upperBound = np.array(value[3:6])*255.0

    # this gives you the mask for those in the ranges you specified,
    # but you want the inverse, so we'll add bitwise_not...
    image_inrange = cv2.inRange(sobel_8u, lowerBound, upperBound)
    image_inrange_color = cv2.bitwise_not(image_bottom, mask=image_inrange)


    features = image.flatten()[:500]
    debug_image.append(image_bottom)
    debug_image.append(sobel_8u)
    debug_image.append(image_inrange)
    debug_image.append(image_inrange_color)
    debug_image.append(quant)
    debug_image.append(image_bottom_lab_res)
    """

    x = []
    y = []
    for x_ in xrange(mask_pasto.shape[1]):
        for y_ in xrange(10, mask_pasto.shape[0]):
            if mask_pasto[y_,x_] == 255:
                x.append(x_)
                y.append(y_)
                break



    x = np.array(x)
    y = np.array(y)

    x1 = x[x<(mask_pasto.shape[1]/2)]
    y1 = y[x<(mask_pasto.shape[1]/2)]
    x1 = x1[y1>(mask_pasto.shape[0]/2)]
    y1 = y1[y1>(mask_pasto.shape[0]/2)]
    y1 = y1.max() - y1
    x = x-x[0] # x offset

    mask_pasto2 = np.zeros(thresh.shape, np.uint8)
    for i in xrange(len(x1)):
        mask_pasto2[y1[i], x1[i]] = 255
    debug_image.append(mask_pasto2)



    x2 = x[x>(mask_pasto.shape[1]/2)]
    y2 = y[x>(mask_pasto.shape[1]/2)]
    y2 = y2 - y2.max()


    from lmfit.models import ExponentialModel

    exp_mod = ExponentialModel(prefix='exp_')
    pars = exp_mod.guess(y1, x=x1)

    out = exp_mod.fit(y, pars, x=x)

    print "error en amplitud:", out.params['exp_amplitude'].stderr,
    print "error en decay:", out.params['exp_decay'].stderr,

    features = np.array([out.best_values['exp_decay'], out.params['exp_decay'].stderr])

    #print(out.fit_report(min_correl=0.5))

    #print pars

    if return_image:
        return features, debug_image
    return features

"""
def calculate_feature(image_path, value=[1,0,0,0,1,0], return_image=False):
    debug_image = []
    vmin = 0
    k = 0

    img = Image.open(image_path).convert("RGB")

    half = 0.5
    img = img.resize([int(half * s) for s in img.size])
    dimy, dimx = img.size

    img = np.array(img)
    img = img[int(0.7 * dimx):dimx, :]

    dimx, dimy, dimz = np.shape(img)
    nimg = np.size(1)

    desp = np.zeros((nimg, MAXDIM))

    # Create Gabor filters
    sig = 2
    sf = 0.2
    gabor45_odd = createGabor(5 * sig - 1, 45, sf, sig, 0)
    gabor0_odd = createGabor(5 * sig - 1, 0, sf, sig, 0)
    gabor135_odd = createGabor(5 * sig - 1, 135, sf, sig, 0)
    gabor45_even = createGabor(5 * sig - 1, 45, sf, sig, np.pi / 2)
    gabor0_even = createGabor(5 * sig - 1, 0, sf, sig, np.pi / 2)
    gabor135_even = createGabor(5 * sig - 1, 135, sf, sig, np.pi / 2)

    # Create Gradient detector filter
    scharr = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],
                       [-10 + 0j, 0 + 0j, +10 + 0j],
                       [-3 + 3j, 0 + 10j, +3 + 3j]])

    imm = image_path

    #print 'Processing ', imm

    # Load and crop images
    # img = Image.open(imm).convert("RGBA")
    img = Image.open(imm).convert("RGB")
    img = img.resize([int(half * s) for s in img.size])
    # print "Loading ", imm, ' size: ', img.size


    # Equalize contrast
    # img = myEqualize(img)
    # img.show()

    # Filter image
    red, green, blue = img.split()

    red = np.array(red)
    blue = np.array(blue)
    green = np.array(green)

    img_flt = np.array(img)

    dimx, dimy, dimz = np.shape(img_flt)
    img_flt = img_flt[int(0.7 * dimx):dimx, :, :]

    red = red[int(0.7 * dimx):dimx, :]
    blue = blue[int(0.7 * dimx):dimx, :]
    green = green[int(0.7 * dimx):dimx, :]

    dimx, dimy, dimz = np.shape(img_flt)
    # print dimx, dimy

    # red, green, blue, alpha = img.split()
    imgCR = scipy.ndimage.filters.gaussian_filter(red, 1, mode='reflect', cval=0.0)
    imgCG = scipy.ndimage.filters.gaussian_filter(blue, 1, mode='reflect', cval=0.0)
    imgCB = scipy.ndimage.filters.gaussian_filter(green, 1, mode='reflect', cval=0.0)
    imgCC = scipy.ndimage.filters.gaussian_filter(red*value[0] + green*value[1] + blue*value[2], 1, mode='reflect', cval=0.0)
    debug_image.append(imgCC)

    imgSR = scipy.ndimage.filters.gaussian_filter(red, 3, mode='reflect', cval=0.0)
    imgSB = scipy.ndimage.filters.gaussian_filter(blue, 3, mode='reflect', cval=0.0)
    imgSG = scipy.ndimage.filters.gaussian_filter(green, 3, mode='reflect', cval=0.0)
    imgSC = scipy.ndimage.filters.gaussian_filter(red*value[3] + green*value[4] + blue*value[5], 3, mode='reflect', cval=0.0)
    debug_image.append(imgSC)

    #imgR_G = imgCR - imgSG
    #imgG_R = imgCG - imgSR
    imgC_C = imgCC - imgSC

    #imgB_Y = imgCB - 0.5 * (imgSG + imgSR)
    #imgY_B = 0.5 * (imgCG + imgCR) - imgSB

    # Convolve with Gabor kernels
    '''
    im_left = scipy.ndimage.filters.convolve(imgR_G,gabor45_odd)**2 +  scipy.ndimage.filters.convolve(imgR_G,gabor45_even)**2
    im_cent = scipy.ndimage.filters.convolve(imgR_G,gabor0_odd)**2 + scipy.ndimage.filters.convolve(imgR_G,gabor0_even)**2
    im_right = scipy.ndimage.filters.convolve(imgR_G,gabor135_odd)**2 + scipy.ndimage.filters.convolve(imgR_G,gabor135_odd)**2

    im_left = signal.convolve2d(imgR_G,gabor45_odd,boundary='symm', mode='full')**2 +  signal.convolve2d(imgR_G,gabor45_even,boundary='symm', mode='full')**2
    '''

    #im_cent = signal.convolve2d(imgR_G, gabor0_odd, boundary='symm', mode='full') ** 2 + signal.convolve2d(imgR_G,
    #                                                                                                       gabor0_even,
    #                                                                                                       boundary='symm',
    #                                                                                                       mode='full') ** 2

    #im_right = signal.convolve2d(imgR_G, gabor135_odd, boundary='symm', mode='full') ** 2 + signal.convolve2d(imgR_G,
    #                                                                                                          gabor135_even,
    #                                                                                                          boundary='symm',
    #                                                                                                          mode='full') ** 2
    #im_left = np.absolute(signal.convolve2d(imgR_G, scharr, boundary='symm', mode='same'))
    #im_left = (im_left - np.min(np.min(im_left))) / (np.max(np.max(im_left) - np.min(np.min(im_left))))

    im_left = np.absolute(signal.convolve2d(imgC_C, scharr, boundary='symm', mode='same'))
    im_left = (im_left - np.min(np.min(im_left))) / (np.max(np.max(im_left) - np.min(np.min(im_left))))
    debug_image.append(im_left)


    desp = getMaxActivated(im_left, MAXDIM)

    for i in xrange(len(debug_image)):
        debug_image[i] = cv2.resize(debug_image[i], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if return_image:
        return (desp, debug_image)

    return desp
"""

if __name__ == "__main__":
    print get_feature("/home/linus/innovaxxion/NEAT/python/examples/images/data/nodulo/20160721014112_0_17_full_image.jpg")
