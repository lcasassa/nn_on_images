import cv2
import numpy as np
import os


# cv2 image to numpy array for input to a net
def get_input(image, return_image=False):
    #output_data os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
    #return [outputsClass.index(output_data)]
    debug_image = []

    image_bottom = image[int(0.7*image.shape[0]):image.shape[0],:,:]

    image_bottom_lab = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2LAB)
    image_lab_a = image_bottom_lab[:, :, 1]
    #image_lab_b = image_bottom_lab[:, :, 2]
    debug_image.append(image_lab_a)
    #debug_image.append(image_lab_b)

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

    mask_cobre = np.zeros(thresh.shape, np.uint8)
    cv2.drawContours(mask_cobre, [biggest_area], 0, 255, 3)
    #debug_image.append(mask_pasto)


    if return_image:
        return mask_cobre, debug_image
    else:
        return mask_cobre


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
    image = cv2.imread(image_path)

    from detector_borde_plastico import detector_borde_plastico
    image_bp, images_dp_debug = detector_borde_plastico(image)
    debug_image.extend(images_dp_debug)


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