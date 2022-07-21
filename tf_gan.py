import random

import loadfunctions as lf
from PIL import Image
import numpy as np


def get_image(np_img:np.ndarray):
    if np_img.mean() > 1:
        return Image.fromarray((np_img).astype(np.uint8))
    else:
        return Image.fromarray(((np_img+1.0) * 128).astype(np.uint8))


def create_x(input_height, input_width, augs, cycles):
    source_images_list = lf.get_image_list(project_dir="source_images")
    resized_images = lf.resize_and_load_image_list(source_images_list, input_width, input_height)
    print(f'done resizing, shape: {resized_images.shape}')
    augmented_images = lf.easy_training_aug(resized_images, num_aug=augs)
    print(f'done aug, shape: {augmented_images.shape}')
    for i in range(cycles):
        augmented_images = np.concatenate([augmented_images,
                                           lf.cycle_hue_of_np_image_array(augmented_images)])
        print(f'done with augmentation', augmented_images.shape)

    print(f'number of augmented images for training: {len(augmented_images)}')
    return augmented_images, resized_images

def create_image_grid(np_img_list:np.ndarray, num_cols:int):
    input_height, input_width = np_img_list[0].shape[:2]
    num_rows = int(np_img_list.shape[0] / num_cols)
    num_rows = num_rows if num_rows * num_cols == np_img_list.shape[0] else num_rows + 1
    img_grid = np.zeros(shape=(num_rows * input_height, num_cols * input_width, 3))
    for i in range(num_rows):
        for j in range(num_cols):
            try:
                img_grid[i * input_height:(i + 1) * input_height, j * input_width:(j + 1) * input_width, :] = \
                    np_img_list[i * num_cols + j, :, :, :]
            except IndexError as e:
                pass

    return img_grid

def create_batch(input_height, input_width,  cycles):
    if "init_done" not in dir(create_batch):
        create_batch.init_done = False

    if not create_batch.init_done:
        print("create_batch running init")
        create_batch.source_image_list = lf.get_image_list(project_dir="source_images")
        create_batch.init_done = True
        create_batch.resized_np_images = lf.resize_and_load_image_list(create_batch.source_image_list, input_width,
                                                                       input_height)
        create_batch.resized_np_images_x2 = lf.resize_and_load_image_list(create_batch.source_image_list,
                                                                          input_width * 2, input_height * 2)
        print(f'done with resize, shape: {create_batch.resized_np_images.shape}')

    target_shape = list(create_batch.resized_np_images.shape)
    target_shape[0] = 0
    augmented_images = np.zeros(shape=target_shape)
    # A-> [A,Aa] -> [A,Aa,Ac,Aac] -> [A,Aa,Ac,Aac]+[Aa,Aaa,Aca,Aaca] = [A,Aa,Ac,Aac,Aa,Aaa,Aca,Aaca]
    print(augmented_images.shape)

    out = []
    for i in range(cycles):
        #out.append(lf.cycle_hue_of_np_image_array(create_batch.resized_np_images.copy()))

        out.append(lf.rotate_images(create_batch.resized_np_images_x2.copy(),input_height,input_width,random.randint(0,359)))
        print(f'done with aug1', out[-1].shape)
        out.append(lf.cycle_hue_of_np_image_array_tf(out[-1]))
        print(f'done with aug2', out[-1].shape)

        #print("after rotation shape: ",out[-1].shape)
        # aug3_images =  lf.rotate_images(aug2_images.copy())
        # augmented_images = np.concatenate([augmented_images,
        #                                   aug1_images,
        #                                   aug2_images,
        #                                   aug3_images])
        augmented_images = np.concatenate(out)
        #print(f'done with aug2', augmented_images.shape)
        #print(f'num unique: {np.unique(augmented_images, axis=0).shape}')
    out.append(create_batch.resized_np_images)
    print(f'shape of out: {out[0].shape}')
    augmented_images = np.concatenate(out)
    print(f'number of augmented images for training: {len(augmented_images)}')
    print(f'num unique: {np.unique(augmented_images, axis=0).shape}')
    np.random.shuffle(augmented_images)
    return augmented_images
