import cv2.ximgproc
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

def match_walls(bimg):

    fig, ax = plt.subplots()
    fft2 = fftpack.fft2(bimg)
    ax.imshow(np.log10(abs(fft2)))
    return

    im_w, im_h = bimg.shape[1], bimg.shape[0]
    pattern = np.ones((11, 9), np.uint8) * 255
    pattern[0, :] = 0
    pattern[-1, :] = 0
    pattern[:, 0] = 0
    pattern[:, -1] = 0
    w, h = pattern.shape[::-1]
    w2, h2 = w//2, h//2
    bimg = bimg.copy()
    bimg[bimg == 0] = 255
    bimg[bimg == 1] = 0
    res = np.zeros_like(bimg, np.float64)
    res[h2:im_h-h2, w2:im_w-w2] = cv2.matchTemplate(bimg, pattern, cv2.TM_CCOEFF_NORMED)

    pattern = pattern.T
    res[w2:im_h-w2, h2:im_w-h2] += cv2.matchTemplate(bimg, pattern, cv2.TM_CCOEFF_NORMED)
    res[res < 0.6] = 0
    ax.imshow(res)




def detect_walls(bimg):
    bimg = bimg.copy()
    bimg[bimg == 0] = 255
    bimg[bimg == 1] = 0
    skeleton = cv2.ximgproc.thinning(bimg)
    return skeleton