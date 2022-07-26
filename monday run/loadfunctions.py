from time import time as time
from PIL import Image
import os, sys, glob
import numpy as np
import random
import tensorflow as tf
import colorsys
from multiprocessing import Pool
from functools import lru_cache


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

    rgb = rgb.astype('float')
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
    hi = hi.astype('uint8')
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
        #convert to rgb
        np_im = np.stack((np_im, np_im, np_im), axis=2)
    im = Image.fromarray(np_im)
    im = im.rotate(random.randint(45, 360 - 45))
    np_image[i] = np.asarray(im)


def rotate_images(np_array):
    """
    input: np array of images [batch,height,width,channels]
    output: np array of images [batch,height,width,channels]
    uses a pool of workers to rotate the images in parallel
    the pool updates the np array in place
    """

    pool = Pool(processes=20)
    pool.starmap(rotate_np_image, zip( [np_array]*len(np_array), range(len(np_array))))
    pool.close()
    pool.join()
    return np_array


def easy_training_aug(X: np.array, num_aug=1):
    out_list = []
    for _ in range(num_aug):
        X = rotate_images(X)
        out_list.append(X)
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



def resize_and_load_image_list(image_file_list, width, height) -> np.array:
    for i, item in enumerate(image_file_list):
        if os.path.isfile(item):
            im = resize_file_keep_aspect(height, item, width)

            if "out_np_arr" not in locals():
                print("creating np image array")
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


def cycle_hue_of_np_image_array(np_images: np.array) -> np.array:
    """
    cycles the hue of a numpy array of images. use a pool to modify the numpy array in-place
    """
    pool = Pool(processes=20)
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
