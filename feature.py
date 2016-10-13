import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from PIL import ImageEnhance
import scipy.ndimage
from scipy import signal
import mdp
import os
import json

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

def get_feature_size():
    return MAXDIM


def get_feature(image_path, recalc=False, return_image=True):
    data_path = image_path.rsplit('.', 1)[0] + '.npy'
    if not recalc and os.path.isfile(data_path):
        return np.load(data_path)

    desp = calculate_feature(image_path)#, return_image=True)

    np.save(data_path, desp)
    return desp


def calculate_feature(image_path):
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

    imgSR = scipy.ndimage.filters.gaussian_filter(red, 3, mode='reflect', cval=0.0)
    imgSB = scipy.ndimage.filters.gaussian_filter(blue, 3, mode='reflect', cval=0.0)
    imgSG = scipy.ndimage.filters.gaussian_filter(green, 3, mode='reflect', cval=0.0)

    imgR_G = imgCR - imgSG
    imgG_R = imgCG - imgSR

    imgB_Y = imgCB - 0.5 * (imgSG + imgSR)
    imgY_B = 0.5 * (imgCG + imgCR) - imgSB

    # Convolve with Gabor kernels
    '''
    im_left = scipy.ndimage.filters.convolve(imgR_G,gabor45_odd)**2 +  scipy.ndimage.filters.convolve(imgR_G,gabor45_even)**2
    im_cent = scipy.ndimage.filters.convolve(imgR_G,gabor0_odd)**2 + scipy.ndimage.filters.convolve(imgR_G,gabor0_even)**2
    im_right = scipy.ndimage.filters.convolve(imgR_G,gabor135_odd)**2 + scipy.ndimage.filters.convolve(imgR_G,gabor135_odd)**2

    im_left = signal.convolve2d(imgR_G,gabor45_odd,boundary='symm', mode='full')**2 +  signal.convolve2d(imgR_G,gabor45_even,boundary='symm', mode='full')**2
    '''

    im_cent = signal.convolve2d(imgR_G, gabor0_odd, boundary='symm', mode='full') ** 2 + signal.convolve2d(imgR_G,
                                                                                                           gabor0_even,
                                                                                                           boundary='symm',
                                                                                                           mode='full') ** 2

    im_right = signal.convolve2d(imgR_G, gabor135_odd, boundary='symm', mode='full') ** 2 + signal.convolve2d(imgR_G,
                                                                                                              gabor135_even,
                                                                                                              boundary='symm',
                                                                                                              mode='full') ** 2

    im_left = np.absolute(signal.convolve2d(imgR_G, scharr, boundary='symm', mode='same'))
    im_left = (im_left - np.min(np.min(im_left))) / (np.max(np.max(im_left) - np.min(np.min(im_left))))


    desp = getMaxActivated(im_left, MAXDIM)

    """
    fig, ax = plt.subplots()
    ax.imshow(im_left, cmap='gray')
    for i in range(np.size(desp)):
        xcord = np.floor(desp[i] / dimy)
        ycord = desp[i] % dimy
        # print '\t(', i, ') ', desp[k,i], xcord, ycord
        ax.scatter([ycord], [xcord], marker='x', color='r')
    plt.show()
    """
    return desp


if __name__ == "__main__":
    print get_feature("/home/linus/innovaxxion/NEAT/python/examples/images/data/nodulo/20160721014112_0_17_full_image.jpg")
