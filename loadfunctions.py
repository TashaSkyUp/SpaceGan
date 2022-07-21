from time import time as time
from PIL import Image
import os, sys, glob
import numpy as np
import random
import tensorflow as tf
import colorsys
from multiprocessing import Pool
from functools import lru_cache
import k_means_plus
from tensorflow_addons.image import rotate
import math


def get_image_list(project_dir="/home/jovyan/work/nasa_planet_dataset/"):
    all_images, src_images, resized_images = get_image_lists(project_dir)
    return src_images


def get_image_lists(project_dir="/home/jovyan/work/nasa_planet_dataset/"):
    exts = ['*.tif', '*.jpg', '*.png']
    all_images = [f for ext in exts for f in glob.glob(os.path.join(project_dir, ext))]
    src_images = [f for f in all_images if not 'resized' in f]
    resized_images = [f for f in all_images if 'resized' in f]
    return (all_images, src_images, resized_images)


def rgb2hsv(rgb):
    """ convert RGB to HSV color space
    ;credit: https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
    :param rgb: np.ndarray
    :return: np.ndarray
    """

    #rgb = rgb.astype('float')
    rgb = tf.cast(rgb,tf.float64)
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def hsv2rgb(hsv):
    """ convert HSV to RGB color space
    ;credit: https://stackoverflow.com/questions/2612361/convert-rgb-values-to-equivalent-hsv-values-using-python
    :param hsv: np.ndarray
    :return: np.ndarray
    """

    hi = np.floor(hsv[..., 0] / 60.0) % 6
    
    #hi = hi.astype('uint8')
    hi = tf.cast(hi,tf.uint8)
    
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
                      
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb


def rotate_np_image(np_image: np.array, i: int):
    np_im = np_image[i, :, :, :].astype('uint8')

    # check if the image is grayscale and convert to rgb
    if np_im.shape.__len__() == 2:
        np_im = np.stack((np_im, np_im, np_im), axis=2)
    # check if image has an alpha channel and remove it if it does
    if np_im.shape.__len__() == 3 and np_im.shape[2] == 4:
        np_im = np_im[:, :, :3]
    if np_im.shape.__len__() == 3 and np_im.shape[2] == 1:
        # convert to rgb
        np_im = np.stack((np_im, np_im, np_im), axis=2)

    im = Image.fromarray(np_im)
    im = im.rotate(random.randint(45, 360 - 45), expand=False)
    im = np.asarray(im)
    im = np.expand_dims(im, axis=0)
    return im


##https://stackoverflow.com/a/50906602
@tf.function
def _rotate_and_crop(images, output_height, output_width, rotation_degree):
    # Rotate the given image with the given rotation degree
    #from tensorflow_addons.image import rotate
    #with tf.device("/gpu:0"):
    if rotation_degree != 0:

        images = rotate(images, math.radians(rotation_degree), interpolation='BILINEAR')
        #rects = [_largest_rotated_rect(i.shape[1],
        #                               i.shape[0],
        #                               math.radians(rotation_degree)
        #                              ) for i in images]
        # fix this to deal with recieving many images.
        #lrr_width, lrr_height = _largest_rotated_rect(int(output_height / 1), int(output_width / 1),
        #                                              math.radians(rotation_degree))

        lrr_width, lrr_height = _largest_rotated_rect(images.shape[1],images.shape[0],
                                                      math.radians(rotation_degree))


        frac = float(lrr_height) / output_height
        # print(rotation_degree)
        # print(image.shape)
        # print(lrr_width, lrr_height)
        # print(frac)
        #if frac < 1:
        #    raise ValueError(f'source to small to rotate: {rotation_degree} degrees')
        cropped_image = tf.image.central_crop(images, .875/frac)

        out_image = tf.image.resize(cropped_image, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR)
        #out_image = images

    return tf.cast(out_image,tf.float64)

#@tf.function
def _rotate_and_crop_many(images, output_height, output_width, rotation_degree):
        def my_rotate_and_crop(a,b):
            #print (a.shape,b.shape)
            return _rotate_and_crop(b,output_height,output_width,rotation_degree)
            #return b
        ini = np.zeros(shape=(output_height,output_width,3))
        #print (ini.shape)
        #print (images.shape)
        return tf.scan(my_rotate_and_crop,images,ini)
    #return _rotate_and_crop(images,output_height,output_width,rotation_degree)
    #return [_rotate_and_crop(i,output_height,output_width,rotation_degree) for i in images]

## https://stackoverflow.com/a/50906602
def _largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow
    Converted to Python by Aaron Snoswell
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    #print(f'finding for: {w}, {h}, {angle}')
    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def rotate_images(np_array, height, width, angle):

    #pool = Pool(processes=10)
    #lenofarray = np_array.shape[0]
    #print(lenofarray)
    ##args =  zip([np_array] * lenofarray,[height] * lenofarray,[width] * lenofarray,range(lenofarray))
    #args = [ (np_array[i,:,:,:],height,width,np.random.randint(1,179)) for i in range(lenofarray)]
    results = _rotate_and_crop_many(np_array,height,width,np.random.randint(1,179))

    #args = [(np_array[i, :, :, :], height, width, 0) for i in range(lenofarray)]
    #print(f'args like: {args[0]} of length {len(args)}')
    #results = pool.starmap(_rotate_and_crop,args)
    #pool.close()
    #pool.join()
    #print(f'results like: {results[0]} of length {len(results)}')
    # target_shape = list(np_array.shape)
    # target_shape[0] = 0
    # augmented_images= np.zeros(shape=target_shape)
    return results
    #return np.concatenate(results)
    #return np_array


def easy_training_aug(X: np.array, num_aug=1, return_original=True):
    out_list = []
    if return_original:
        for _ in range(num_aug):
            X = rotate_images(X)
            out_list.append(X)
            out_list.append(np.flip(X, 3))
            out_list.append(np.flip(X, 1))
            out_list.append(np.flip(X, 2))
            out_list.append(np.flip(np.flip(X, 2), 1))
            out_list.append(np.flip(np.flip(X, 2), 2))

    else:
        for _ in range(num_aug):
            X = rotate_images(X)
            out_list.append(np.flip(X, 3))
            out_list.append(np.flip(X, 1))
            out_list.append(np.flip(X, 2))
            out_list.append(np.flip(np.flip(X, 2), 1))
            out_list.append(np.flip(np.flip(X, 2), 2))

    return np.array(out_list).reshape(
        (-1, X.shape[1], X.shape[2], X.shape[3]))


def resize_file_list(files, outdir, width=512, height=384) -> list[np.array]:
    print(files, outdir, type(files))
    output_list = []
    if isinstance(files, str):
        files = [files]

    for item in files:
        print(item)

        base, file_name = os.path.split(os.path.abspath(item))
        file_no_ext, ext = os.path.splitext(file_name)

        if os.path.isfile(item):
            new_im = resize_file_keep_aspect(height, item, width)
            # new_im.show()
            np_arr = np.asarray(new_im)

            # print(np_arr.shape)

            out_full_file_name = file_no_ext + ' resized.png'
            out_full_file_name = os.path.join(base, "..", outdir, out_full_file_name)
            out_full_file_name = os.path.abspath(out_full_file_name)

            # output_list.append(
            #    np_arr.reshape(height,width,3)
            # )

            output_list.append(out_full_file_name)
            re_pil = tf.keras.preprocessing.image.array_to_img(np_arr)
            re_pil.save(out_full_file_name)

    return output_list


def resize_file_keep_aspect(height, item, width) -> Image:
    im = Image.open(item)
    new_im = resize_image_keep_aspect(height, im, width)
    return new_im


def resize_image_keep_aspect(height, im, width):
    new_im = Image.new("RGB", (width, height))
    ar = im.width / im.height
    # print(ar)
    if ar >= (width / height):
        new_width = int(height * ar)
        new_height = height
        padx = int((new_width - width) / 2)
        pady = 0
    else:
        new_height = int(width / ar)
        new_width = width
        pady = int((new_height - height) / 2)
        padx = 0
    imResize = im.resize((new_width, new_height), Image.LANCZOS)
    # imResize.show()
    crop_width = (new_width - padx - padx)
    crop_height = (new_height - pady - pady)
    x_mod = width - crop_width
    y_mod = height - crop_height
    # print(x_mod,y_mod,new_width,new_height,padx,pady)
    imResize = imResize.crop((
        padx, pady,
        new_width - padx + x_mod,
        new_height - pady + y_mod
    ))
    # imResize.show()
    new_im.paste(imResize)
    return new_im


def load_image_list_to_np(image_file_list):
    for i, item in enumerate(image_file_list):
        if os.path.isfile(item):
            im = Image.open(item)
            if "out_np_arr" not in locals():
                print("creating np image array")
                out_np_arr = np.zeros((len(image_file_list), im.height, im.width, 3))
                # print (out_np_arr.shape)
            np_im = np.asarray(im)
            # print(np_im.shape)
            out_np_arr[i] = np_im
            # print(out_np_arr.shape)
    return out_np_arr


def rotate_image_list(images):
    """
    Rotates a list of images.
    """
    for i, item in enumerate(np_images):
        if os.path.isfile(item):
            im = Image.open(item)
            im = im.rotate(random.randint(0, 360))
            np_images[i] = np.asarray(im)
    return np_images


def get_centroid_groups(C, S):
    groups = [[v] for v in C]
    for centroid in range(len(C)):
        current_cluster = [v["i"] for v in S if v["c"] == centroid]
        groups[centroid] = groups[centroid] + current_cluster
    return groups


def get_score_for_num_centroids(X, num_centroids):
    C, S = k_means_plus.compute(X, num_centroids)
    groups = get_centroid_groups(C, S)
    lens_of_groups = [len(v) for v in groups]
    score = (num_centroids / (max(lens_of_groups) - min(lens_of_groups))) / num_centroids

    median_length_of_group = np.median(lens_of_groups)
    perfect_mean = len(X) / len(lens_of_groups)
    score = score + (median_length_of_group / perfect_mean)
    return (score, groups)


def split_image_np_list_into_k_means_groups(images, k, verbose=1):
    """
    Splits a list of images into k groups.
    """
    C, S = k_means_plus.compute(images, k)
    return get_centroid_groups(C, S)


def get_best_k_for_images(images, k_min, k_max, verbose=1):
    """
    Gets the best k for a list of images.
    """
    best_k = k_min
    best_score = 0
    results = []
    for k in range(k_min, k_max):
        score, groups = get_score_for_num_centroids(images, k)
        results.append((k, score, groups))
        if score > best_score:
            best_score = score
            best_k = k
        if verbose == 1:
            print("k:", k, "score:", score)
            for group in groups:
                print(len(group))
                if verbose > 1:
                    print(group)

    return best_k, results


def resize_and_load_image_list(image_file_list, width, height) -> np.array:
    for i, item in enumerate(image_file_list):
        if os.path.isfile(item):
            im = resize_file_keep_aspect(height, item, width)

            if "out_np_arr" not in locals():
                print("creating resized np image array")
                out_np_arr = np.zeros((len(image_file_list), im.height, im.width, 3))
                # print (out_np_arr.shape)
            np_im = np.asarray(im)
            # check if the image is grayscale and convert to rgb
            if np_im.shape.__len__() == 2:
                np_im = np.stack((np_im, np_im, np_im), axis=2)
            # check if image has an alpha channel and remove it if it does
            if np_im.shape.__len__() == 3 and np_im.shape[2] == 4:
                np_im = np_im[:, :, :3]

            out_np_arr[i] = np_im
        else:
            raise InvalidArugmentException()
    return out_np_arr


def load_rotate_and_save_image_list(lt: [], outdir: str):
    """load a list of image files, rotates them, and saves them with their original filename but with  'rotated'
    appended, and saves them to the outdir directory. and creates outdir and creates outdir if it doesn't exist. """
    if not os.path.exists(outdir):
        os.makedirs(os.path.join(outdir, ".."))

    for i, item in enumerate(lt):
        if os.path.isfile(item):
            im = Image.open(item)
            im = im.rotate(random.randint(20, 360))
            np_im = np.asarray(im)
            base, file_name = os.path.split(os.path.abspath(item))
            file_no_ext, ext = os.path.splitext(file_name)
            out_full_file_name = file_no_ext + '_rotated.png'
            out_full_file_name = os.path.join(base, "..", outdir, out_full_file_name)
            out_full_file_name = os.path.abspath(out_full_file_name)
            # check to see if image is gray scale
            if len(np_im.shape) == 2:
                np_im = np_im.reshape(np_im.shape[0], np_im.shape[1], 1)
            re_pil = tf.keras.preprocessing.image.array_to_img(np_im)
            re_pil.save(out_full_file_name)
    return lt


def load_and_augment_image_list(list: []) -> np.array:
    np_images = load_image_list_to_np(resized_image_list)
    np_resized_images = resized_image_list

    print(np_images[0], np_images.shape, np_images[0].shape)
    np_images = easy_training_aug(np_images)
    return np_images


def np_image_array_to_image_list(np_images: np.array) -> []:
    """np_images is an numpy array of images with shape (-1,hieght,width,channels)
    iterates over np_images converting each image to a image and returns a list.
    uses a pool to modify the numpy array in-place
    """
    pool = Pool(processes=10)
    my_map = map(np_images.__getitem__, range(len(np_images)))
    my_map = [i.astype(np.uint8) for i in my_map]
    image_list = pool.map(Image.fromarray, my_map)
    return image_list


def save_np_image_list(list_of_np_images, outdir: str):
    image_list = np_image_array_to_image_list(list_of_np_images)
    save_image_list(image_list, outdir)


def save_image_list(list_of_images: [Image], outdir: str):
    """
    saves a list of images to the outdir directory use a pool.
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pool = Pool(processes=10)
    my_map = pool.starmap(Image.Image.save,
                          zip(list_of_images,
                              [os.path.join(outdir, str(i)) + ".png" for i in range(len(list_of_images))]))

    return list_of_images


def cycle_hue_of_image(image: Image) -> Image:
    """
    cycles the hue of an image.
    """
    im = image.convert('HSV')
    data = np.array(im)
    data[..., 0] = (data[..., 0] + 1) % 360
    im = Image.fromarray(data)
    im = im.convert('RGB')
    return im


def cycle_hue_of_image_list(image_list: [Image]) -> [Image]:
    """
    cycles the hue of a list of images.
    """
    for i, item in enumerate(image_list):
        image_list[i] = cycle_hue_of_image(item)
    return image_list


def cycle_hue_of_image(im):
    """
    cycles the hue of an image by 20 degrees.
    """
    im = im.convert('HSV')
    data = np.array(im)
    data[..., 0] = (data[..., 0] + 20) % 360
    im = Image.fromarray(data)
    im = im.convert('RGB')
    return im


def cycle_hue_of_np_image(image: np.array, degrees=None) -> np.array:
    """
    cycles the hue of a numpy image by degrees.
    """
    if degrees == None:
        degrees = random.randint(-30, 30)

    image = rgb2hsv(image)
    image[..., 0] = (image[..., 0] + degrees) % 360
    image = hsv2rgb(image)
    return image


def cycle_hue_of_np_image_array_tf(images):    
    def tmp_cycle(a,b):        
        return cycle_hue_of_np_image(b,degrees=None)
    
    return tf.scan(lambda a,b:tmp_cycle(a,b), images)
    
    
def cycle_hue_of_np_image_array(np_images: np.array) -> np.array:
    """
    cycles the hue of a numpy array of images. use a pool to modify the numpy array in-place
    """
    np_images= np.array(np_images)
    from tensorflow.python.ops.numpy_ops import np_config
    np_config.enable_numpy_behavior()
    pool = Pool(processes=10)
    my_map = map(np_images.__getitem__, range(len(np_images)))
    my_map = [i.astype(np.uint8) for i in my_map]
    my_map = pool.map(cycle_hue_of_np_image, my_map)
    return np.asarray(my_map)

    # for i, item in enumerate(np_images):
    #        np_images[i] = cycle_hue_of_np_image(item,random.randint(-30,+30))
    # return np_images


if __name__ == "__main__":
    # time these calls

    start = time()
    source_image_list = get_image_list("source_images")
    end = time()
    print("get_image_list took: ", end - start)
    start = time()
    np_images = resize_and_load_image_list(tuple(source_image_list), width=256, height=int(256 / (16 / 10)))
    end = time()
    print("resize_and_load_image_list took: ", end - start)
    start = time()
    np_augmented_images = easy_training_aug(np_images, num_aug=3)
    end = time()
    print("easy_training_aug took: ", end - start)
    np_augmented_images = cycle_hue_of_np_image_array(np_augmented_images)
    start = time()
    image_list = np_image_array_to_image_list(np_augmented_images)
    end = time()
    print("np_image_array_to_image_list took: ", end - start)
    start = time()
    save_image_list(image_list, "augmented_images")
    end = time()
    print("save_image_list took: ", end - start)


def train(dataset, epochs, start_epoch=0):
    results_history = {}
    # if start_epoch == -1 and results_history exists on the file system load results_history
    if start_epoch == -1 and os.path.exists('results_history.pkl'):
        with open('results_history.pkl', 'rb') as f:
            results_history = pickle.load(f)
            print("loaded results_history")
        start_epoch = results_history['epoch']
    elif start_epoch == -1:
        results_history = {}

    for epoch in range(start_epoch, epochs):
        results_history["epoch"] = epoch
        dict_of_tots = {"disc_loss_epoch_tot": 0,
                        "gen_loss_epoch_tot": 0,
                        "real_loss_epoch_tot": 0,
                        "fake_loss_epoch_tot": 0,
                        "fooled_epoch_tot": 0}

        start = time.time()
        tot_batches = 0
        # train on each batch and generates/updates results history
        for image_batch in dataset:
            # accumulate the results of each train_step
            results = train_step(image_batch)
            # if results_history is empty, initialize it with the first results
            if len(results_history) == 1:
                results_history = dict(results)
                for key in results_history:
                    if key != 'epoch':
                        results_history[key] = []

            else:
                # append results to results_history
                for key in results_history:
                    if key != 'epoch':
                        results_history[key].append(results[key])
            # accumulate the results of each train_step
            for key in dict_of_tots:
                dict_of_tots[key] += results[key]
            tot_batches += 1

        display(dict_of_tots, epoch, epochs, results_history, tot_batches)

        # save results_history to disk as results_history.pkl
        with open('results_history.pkl', 'wb') as f:
            pickle.dump(results_history, f)

        print("Time for epoch: ", time.time() - start)


def display(dict_of_tots, epoch, epochs, results_history, tot_batches):
    # Produce images for the GIF as you go
    display.clear_output(wait=False)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)
    # use matplotlib to plot line plots of the losses over time
    colors_bright = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']
    # define some darker colors
    colors_dark = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                   '#17becf']
    # make the plot wide
    plt.figure(figsize=(10, 4))
    for i, key in enumerate(results_history):
        if (key != 'epoch') & (key != 'fooled'):
            plot_history_clipped = prepare_series_for_graph(results_history[key], clip=True, smooth=False)
            plot_history_clipped_smoothed = prepare_series_for_graph(results_history[key], clip=True, smooth=True)
            plt.plot(plot_history_clipped, color=colors_dark[i], linewidth=1, alpha=.5)
            # use a thinner line for the smoothed results_history
            plt.plot(plot_history_clipped_smoothed, label=key + ' smoothed', color=colors_bright[i], linewidth=2,
                     alpha=1.0)
    plt.legend()
    plt.show()
    plot_history_clipped = prepare_series_for_graph(results_history["fooled"], clip=True, smooth=False)
    plot_history_clipped_smoothed = prepare_series_for_graph(results_history["fooled"], clip=True, smooth=True)
    plt.figure(figsize=(10, 4))
    plt.plot(plot_history_clipped, color=colors_dark[i], linewidth=1, alpha=.5)
    # use a thinner line for the smoothed results_history
    plt.plot(plot_history_clipped_smoothed, label="fooled" + ' smoothed', color=colors_bright[i], linewidth=2,
             alpha=1.0)
    plt.legend()
    plt.show()
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    # print the results average using f strings
    print(f"Epoch: {epoch + 1}/{epochs}")
    print(f"Discriminator loss: {dict_of_tots['disc_loss_epoch_tot'] / tot_batches}")
    print(f"Generator loss: {dict_of_tots['gen_loss_epoch_tot'] / tot_batches}")
    print(f"Real loss: {dict_of_tots['real_loss_epoch_tot'] / tot_batches}")
    print(f"Fake loss: {dict_of_tots['fake_loss_epoch_tot'] / tot_batches}")
    print(f"Fooled: {dict_of_tots['fooled_epoch_tot'] / tot_batches}")


if __name__ == "__mainn__":
    print(__file__)
    source_image_list = get_image_list("source_images")

    with Pool(10) as p:
        map_list = list(zip(source_image_list, ["training_images"] * len(source_image_list)))
        res = p.starmap(resize_file_list, map_list)
        resized_image_list = [i[0] for i in res]
        print(resized_image_list)

        # resized_image_list = resize_file_list(source_image_list,"training_images")

    np_images = load_image_list_to_np(resized_image_list)
    print(np_images[0], np_images.shape, np_images[0].shape)
    np_images = easy_training_aug(np_images)
