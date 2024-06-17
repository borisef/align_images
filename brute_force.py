import math
import numpy
import numpy as np
import scipy.ndimage as ndimage
import imagecodecs
import myimreg
import cv2
from matplotlib import pyplot, patches


def rotate_img_around_point_and_shift(img1, angle , ref_point , txty, out_shape, show = True,
                                      fig_title = "rotate_img_around_point_and_shift"):

     T_ref = myimreg.similarity_matrix(scale=1, angle=0, vector=(-ref_point[0], -ref_point[1]))
     H = myimreg.similarity_matrix(scale=1, angle=angle, vector=(0, 0))
     T_back = myimreg.similarity_matrix(scale=1, angle=0, vector=(ref_point[0], ref_point[1]))
     T_last =  myimreg.similarity_matrix(scale=1, angle=0, vector=(txty[0], txty[1]))

     H_temp = np.matmul(T_last,np.matmul(T_back,np.matmul(H,T_ref)))

     warped3 = cv2.warpPerspective(img1, H_temp, out_shape, \
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
     if(show):
         pyplot.figure(fig_title)
         pyplot.imshow(warped3*0.5)
     return warped3






def cosine_diff(a,b):
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    a = a - a.mean()
    b = b - b.mean()
    sim = np.dot(a,b) / np.sqrt(np.dot(a,a)*np.dot(b,b)+ 0.00001)
    #print(sim)
    return np.abs(sim)

def imshow_dif(im0, im1, im2,cmap=None, fig_title = "imshow_dif",**kwargs):
    """Plot images using matplotlib."""
    from matplotlib import pyplot

    pyplot.figure(fig_title)
    pyplot.subplot(221)
    pyplot.imshow(im0, cmap, **kwargs)
    pyplot.subplot(222)
    pyplot.imshow(im1, cmap, **kwargs)
    try:
        pyplot.subplot(223)
        pyplot.imshow(im1*0.5 + im0*0.5, cmap, **kwargs)
    except:
        pass
    pyplot.subplot(224)
    pyplot.imshow(im1 * 0.5 + im2 * 0.5, cmap, **kwargs)
    pyplot.show()
    pyplot.pause(0.1)

def transform_show(scale, rotate,translate,img, tar_img = None, show = True,
                   fig_title = "transform_show"):
# The order of transformations is: scale, rotate, translate
# after myimreg.similarity call it with
# transform_show(1/scale,-angle,(-t0,-t1),search)

    h, w = img.shape

    #SCALE
    H = myimreg.similarity_matrix(scale=scale, angle=0, vector=(0, 0))
    warped1 = cv2.warpPerspective(img, H, (w, h), \
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    #ROTATE AROND CENTER OF MAGE
    T = myimreg.similarity_matrix(scale=1, angle=0, vector=(-w/2, -h/2))
    H = myimreg.similarity_matrix(scale=1, angle=rotate, vector=(0, 0))
    T_back = myimreg.similarity_matrix(scale=1, angle=0, vector=(w/2, h/2))
    H = np.matmul(T,np.matmul(H,T_back))
    warped2 = cv2.warpPerspective(warped1, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    H = myimreg.similarity_matrix(scale=1, angle=0, vector=translate)

    if(tar_img is not None):
        h, w = tar_img.shape

    warped3 = cv2.warpPerspective(warped2, H, (w, h), \
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if(show):
        pyplot.figure(fig_title)
        if(tar_img is not None):
            pyplot.imshow(warped3*0.5 + tar_img*0.5)
        else:
            pyplot.imshow(warped3)
    return warped3

def transform_show_consistent(scale, rotate,translate,img, target_img=None, show = True,fig_title = 'const_transform_show'):
# The order of transformations is: scale, rotate, translate
# after myimreg.similarity call it with
# transform_show(1/scale,-angle,(-t0,-t1),search)
    if(target_img is None):
        target_img = np.zeros_like(img)

    t0 = translate[0]
    t1 = translate[1]

    h, w = img.shape
    h_tar, w_tar = target_img.shape


    # if rotate < -90.0:
    #     rotate += 180.0
    # elif rotate > 90.0:
    #     rotate -= 180.0

    im2 = ndimage.zoom(img, 1.0 / scale)
    im2 = ndimage.rotate(im2, rotate, reshape = True)


    if im2.shape < target_img.shape:
        t = numpy.zeros_like(target_img)
        t[: im2.shape[0], : im2.shape[1]] = im2
        im2 = t
    elif im2.shape > target_img.shape:
        im2 = im2[: img.shape[0], : img.shape[1]]


    im2 = ndimage.shift(im2, [t0, t1])


    if(show):
        pyplot.figure(fig_title)
        pyplot.imshow(im2)
    return im2


def recover_transform(img, ref_img_point =None, scale=1, rotate=0, translate=(0, 0),
                      target_img=None, show = True, fig_title = 'recover_transform_show'):
    #ref_img_point - center of ORIGINAL search window (before zoom)
    # translate = center_shift + [orig_t1, orig_t0]
    # rotate = -orig_angle
    h,w = img.shape
    if(ref_img_point is None):
        ref_img_point = np.array([w / 2, h / 2]) #center of img
    if(target_img is None):
        target_img = np.zeros_like(img)

    # scale img
    img1  = ndimage.zoom(img,scale)
    ref_img_point = ref_img_point*scale #scale ref point

    #rotate around the point  and shift
    ht,wt = target_img.shape
    img2 = rotate_img_around_point_and_shift(img1, angle = rotate, ref_point = ref_img_point,txty = translate,  out_shape = (wt,ht), show=False)

    if(show):
        pyplot.figure(fig_title)
        pyplot.imshow(img2)

    return img2





def my_brute_force_scale_invariant_template_matching(
    template,  # grayscale image
    search,  # scaled and cropped grayscale image
    zooms=(1.0, 1.1, 0.9),  # sequence of zoom factors to try
    size=None,  # power-of-two size of square sliding window
    delta=64, #None,  # advance of sliding windows. default: half window size
    min_overlap=0.5,#0.25,  # minimum overlap of search with window
    max_angle=20,  # small value like 0.5 for no rotation
    min_cos_similarity = 0.7
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

    search = search[:size, :size] # crop max power of 2 window
    best_sim = 0
    best_sim_params = None

    for zoom in zooms:
        windows = numpy.lib.stride_tricks.sliding_window_view(
            ndimage.zoom(template, zoom), search.shape
        )[::delta, ::delta] #prepare sliding windows
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                print('.', end='')
                window = windows[i, j]
                im2, scale, angle, (t0, t1), center_shift, orig_scale, (orig_t0, orig_t1) = myimreg.similarity(window, search)

                #find overlap and compare to threshold
                nz = (im2>0.01).sum()/(im2.shape[0]*im2.shape[1])

                sim = cosine_diff(im2,window)
                if(best_sim < sim and sim >  min_cos_similarity  and max_angle > np.abs(angle) and nz> min_overlap):
                    best_sim = sim
                    # im2, scale, angle, (t0, t1), center_shift, orig_scale, (orig_t0, orig_t1) =\
                    #     myimreg.similarity(window, search)

                    print('best sim:' + str(sim))
                    #imshow_dif(search, window,im2,fig_title="imshow_dif_window")  # crop-good, crop_rot - GOOD

                    best_t0 = (i * delta + orig_t0) / zoom
                    best_t1 = (j * delta + orig_t1) / zoom
                    best_scale = orig_scale/ zoom
                    best_angle = -angle
                    best_sim_params = {'transformed': im2, 'scale': scale, 'best_angle': best_angle, 't0': t0, 't1': t1,
                                       'zoom': zoom, 'i': i, 'j': j, 'center_shift': center_shift,
                                       'best_scale': best_scale, 'best_t0': best_t0, 'best_t1': best_t1
                                       }

                    #(h, w) = search.shape
                    # ref_point = np.array([w/2,h/2]) # center of search
                    # translate = center_shift + [orig_t1, orig_t0]

                    # pyplot.figure('im2');pyplot.imshow(im2)
                    # recover_transform(search, ref_img_point =ref_point, scale=orig_scale, rotate=best_angle, translate=translate,
                    #                   target_img=None, show = True, fig_title = 'recover_transform_show')

                    #pyplot.figure('template');pyplot.imshow(template)
                    (h, w) = search.shape
                    ref_point = np.array([w / 2, h / 2])  # center of search or center of search if search is subimage in larger image
                    translate2template= center_shift + [best_t1, best_t0]
                    search_transformed = recover_transform(search, ref_img_point=ref_point, scale=best_scale, rotate=best_angle,
                                      translate=translate2template,
                                      target_img=template, show=False, fig_title='recover_transform_in_template')

                    pyplot.show()
                    print("OK")


    if(best_sim < min_cos_similarity):
        print('no match of search image found in template')
        return None
    else:
       #imshow_dif(search, template,search_transformed, fig_title="imshow_dif_search_to_template")
       best_sim_params['transformed'] = search_transformed
       print('OK')
    return best_sim_params


def brute_force_scale_invariant_template_matching(
    template,  # grayscale image
    search,  # scaled and cropped grayscale image
    zooms=(1.0, 0.5, 0.25),  # sequence of zoom factors to try
    size=None,  # power-of-two size of square sliding window
    delta=8, #None,  # advance of sliding windows. default: half window size
    min_overlap=0.5,#0.25,  # minimum overlap of search with window
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
            ndimage.zoom(template, zoom), search.shape
        )[::delta, ::delta]
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                print('.', end='')
                window = windows[i, j]
                im2, scale, angle, (t0, t1), center_shift= myimreg.similarity(window, search)
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

if __name__ == "__main__":

    template = imagecodecs.imread('/home/borisef/projects/align_images/im2gray.jpg')
    #search = imagecodecs.imread('/home/borisef/projects/align_images/crop2.jpg')
    # search = imagecodecs.imread('/home/borisef/projects/align_images/crop1.jpg')
    search = imagecodecs.imread('/home/borisef/projects/align_images/crop_and_rot.jpg')
    # search = imagecodecs.imread('/home/borisef/projects/align_images/zoom105.png')
    #search = imagecodecs.imread('/home/borisef/projects/align_images/zoom105rot5.png')#soso

    mr = my_brute_force_scale_invariant_template_matching(
        template, search, zooms=[ 1.0],delta = 32,min_overlap=0.6, max_angle=20.0,min_cos_similarity=0.7
    )
    #vizualize results
    (h, w) = search.shape
    ref_point = np.array([w / 2, h / 2])  # center of search or center of search if search is subimage in larger image
    translate2template = mr['center_shift'] + [mr['best_t1'], mr['best_t0']]
    search_transformed = recover_transform(search, ref_img_point=ref_point, scale=mr['best_scale'], rotate=mr['best_angle'],
                                           translate=translate2template,
                                           target_img=template, show=False, fig_title='recover_transform_in_template')

    imshow_dif(search, template, search_transformed, fig_title="imshow_dif_search_to_template")
    pyplot.show()
    print('Done')