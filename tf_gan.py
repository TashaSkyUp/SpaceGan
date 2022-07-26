import random

import loadfunctions as lf
from PIL import Image
import numpy as np
from IPython import display as ipydisplay
import time
import wandb
import matplotlib.pyplot as plt
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

        out.append(
            lf.rotate_images(create_batch.resized_np_images_x2.copy(),
                             input_height,input_width,
                             random.randint(0,359)))
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
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False).numpy()
  print(predictions.shape,predictions.dtype)
  fig = plt.figure(figsize=(8,2),dpi=160)
  #fig = plt.figure()

  for i in range(predictions.shape[0]):
      plt.subplot(2, 8, i+1)
      plt.imshow((predictions[i, :, :, :] * 255).astype(np.uint8))
      plt.axis('off')  
  plt.show()

def save_images(image = None,noise= None,generator= None,epoch=None,run=None):
    if not image:
        #print("noise: ",noise)
        predictions = generator(noise, training=False).numpy()
        input_height = predictions.shape[1]
        input_width =  predictions.shape[2]
        #print(predictions.shape,predictions.dtype)

        # create hstack the images
        all = [(x,y) for x in range(8) for y in range(2)]

        for i,t in enumerate(all):
            x,y = t
            display.np_img[
                y*input_height:(y+1)*input_height,
                x*input_width:(x+1)*input_width,
                :] = predictions[i,:,:,:]
        img = Image.fromarray(((display.np_img+1)* 128).astype(np.uint8))
    
    else:
        img = image
    
    img.save('predictions_at_epoch_{:04d}.png'.format(epoch+1))
    if run:
        run.log({"predictions":wandb.Image(img)})
    
    return img

def display(models:[],
            epoch,
            epoch_max,
            results_history,
            tot_batches,
            noise,
            plot_graphs=True,
            plot_images=True,
            ):
    
    start = time.time()
    
    # Produce images for the GIF as you go
    ipydisplay.clear_output(wait=False)
    if plot_images:
        t = save_images(noise=noise,generator =models[0],epoch=epoch)        
        ipydisplay.display(t)
        
    if plot_graphs:
        clrs = plt.cm.get_cmap("hsv", len(results_history))
        # make the plot wide
        plt.figure(figsize=(8,2),dpi=160)
        for i, key in enumerate(results_history):
            if ('epoch' not in key) & (key != "fooled")&('_lr' not in key)&('scale' not in key):
                clr = clrs(i)
                clr=[
                     max(clr[0]-(i/255),0),
                     max(clr[1]-(i/255),0),
                     max(clr[2]-(i/255),0)
                 ]
                plot_history_clipped = prepare_series_for_graph(results_history[key][-1000:], clip=True, smooth=False)
                plot_history_clipped_smoothed = prepare_series_for_graph(results_history[key][-1000:], clip=True, smooth=True)

                plt.plot(plot_history_clipped, linewidth=.5, alpha=.35, color=clr)
                # use a thinner line for the smoothed results_history
                plt.plot(plot_history_clipped_smoothed, label=key + ' smoothed', linewidth=1,alpha=.8, color=clr)

        plt.legend(loc= 'upper left')
        plt.show()

        #plot_history_clipped = prepare_series_for_graph(results_history["fooled"][-1000:], clip=True, smooth=False)
        #plot_history_clipped_smoothed = prepare_series_for_graph(results_history["fooled"][-1000:], clip=True, smooth=True)

        plt.figure(figsize=(8,2),dpi=160)
        plt.rcParams.update({'font.size': 5})
        #plt.plot(plot_history_clipped, linewidth=.5, alpha=.33)
        # use a thinner line for the smoothed results_history

        #plt.yscale('log')
        #plt.ylim(0,20)
        #plt.plot(results_history["fooled_per"], label="fooled_per", linewidth=1,alpha =.80)
        plt.plot(results_history["g_grad_scale_accum"][-1000:], label="g_grad_scale_accum" ,  linewidth=1,alpha =1)
        plt.plot(results_history["d_grad_scale_accum"][-1000:], label="d_grad_scale_accum" ,  linewidth=1,alpha =1)
        plt.plot(results_history["equilibrium_scale"][-1000:], label="equilibrium_scale" ,  linewidth=1,alpha =1)
        plt.plot(results_history["grad_scale_power"][-1000:], label="grad_scale_power" ,  linewidth=1,alpha =1)
        plt.legend(loc= 'upper left')
        plt.show()
        
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    # print the results average using f strings

    def mean(lst:[]):
        try:
            return sum(lst) / len(lst)
        except tf.errors.InvalidArgumentError as e:
            return tf.reduce_mean(tf.cast(lst,tf.float32))
        
    if "current_epoch" in results_history:
        results_history["last_epoch"] = results_history['current_epoch']

    results_history['current_epoch']={ k+"_mean":mean(v[-100:]) for k,v in results_history.items() if type(v)==list}

    print(f"gen_loss: {results_history['current_epoch']['gen_loss_mean']}")
    print(f"disc_loss: {results_history['current_epoch']['disc_loss_mean']}")
    print(f"fooled_per: {results_history['current_epoch']['fooled_per_mean']}")
    print(f"g_grad_scale_accum: {results_history['current_epoch']['g_grad_scale_accum_mean']}")
    print(f"d_grad_scale_accum: {results_history['current_epoch']['d_grad_scale_accum_mean']}")
    print(f"Epoch: {epoch + 1} / {epoch_max}")
    print(f"Batches: {tot_batches}")
    print(f"D Time: {time.time() - start} for images: {plot_images} and graphs: {plot_graphs}")
    
def prepare_series_for_graph(np_arr,smooth = True,clip =True):
    # print performance for this function
    s_time = time.time()
    out = np.array(np_arr)

    # smooth the results_history
    if smooth:
        out  = np.convolve(out, np.ones((10,))/10, mode='valid')
    # clip the results_history
    if clip:
        c=np.percentile(out, 95)
        out = out[out < c]
    e_time = time.time()
    #print(f"Time for smooth: {smooth} and clip: {clip} was {e_time - s_time}")

    return out.astype(np.float16)