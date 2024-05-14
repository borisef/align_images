import math
import numpy
import scipy.ndimage
import imagecodecs
import imreg
from matplotlib import pyplot, patches


def brute_force_scale_invariant_template_matching(
    template,  # grayscale image
    search,  # scaled and cropped grayscale image
    zooms=(1.0, 0.5, 0.25),  # sequence of zoom factors to try
    size=None,  # power-of-two size of square sliding window
    delta=None,  # advance of sliding windows. default: half window size
    min_overlap=0.25,  # minimum overlap of search with window
    max_diff=0.05,  # max average of search - window differences in overlap
    max_angle=0.5,  # no rotation
):
    """Return yoffset, xoffset, and scale of first match of search in template.

    Iterate over scaled versions of the template image in overlapping sliding
    windows and run FFT-based algorithm for translation, rotation and
    scale-invariant image registration until a match of the search image is
    found in the sliding window.

    """
    if size is None:
        size = int(pow(2, int(math.log(min(search.shape), 2))))
    if delta is None:
        delta = size // 2
    search = search[:size, :size]
    for zoom in zooms:
        windows = numpy.lib.stride_tricks.sliding_window_view(
            scipy.ndimage.zoom(template, zoom), search.shape
        )[::delta, ::delta]
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                print('.', end='')
                window = windows[i, j]
                im2, scale, angle, (t0, t1) = imreg.similarity(window, search)
                diff = numpy.abs(im2 - window)[im2 != 0]
                if (
                    abs(angle) < max_angle
                    and diff.size / window.size > min_overlap
                    and numpy.mean(diff) < max_diff
                ):
                    return (
                        (i * delta - t0) / zoom,
                        (j * delta - t1) / zoom,
                        1 / scale / zoom,
                    )
    raise ValueError('no match of search image found in template')


def rgb2gray(rgb, scale=None):
    """Return float grayscale image from RGB24 or RGB48 image."""
    scale = numpy.iinfo(rgb.dtype).max if scale is None else scale
    scale = numpy.array([[[0.299, 0.587, 0.114]]], numpy.float32) / scale
    return numpy.sum(rgb * scale, axis=-1)


template = imagecodecs.imread('cw1_IMG_9037.jpg')
search = imagecodecs.imread('cw1_p1_9037_kzw.jpg')

yoffset, xoffset, scale = brute_force_scale_invariant_template_matching(
    rgb2gray(template), rgb2gray(search), zooms=(0.5,)
)
print(yoffset, xoffset, scale)

figure, ax = pyplot.subplots()
ax.imshow(template)
rect = patches.Rectangle(
    (xoffset, yoffset),
    scale * search.shape[1],
    scale * search.shape[0],
    linewidth=1,
    edgecolor='r',
    facecolor='none',
)
ax.add_patch(rect)
pyplot.show()