import os
import pdb

import tensorflow as tf

import numpy as np

import misc
import config

sess = tf.InteractiveSession()

run_id = "karras2018iclr-celebahq-1024x1024.pkl"
random_seed = 1000
result_subdir = "results/random_sample"

network_pkl = misc.locate_network_pkl(run_id, None)
print('Loading network from "%s"...' % network_pkl)
G, D, Gs = misc.load_network_pkl(run_id, None)

#random_state = np.random.RandomState(random_seed)
np.random.seed(1337)

def generate_fake_images(grid_size=[1,1], num_pngs=1, image_shrink=1, minibatch_size=8):
    t = np.linspace(0,3.141592*2,num_pngs)
    latents = np.zeros((1,512), dtype=np.float32)
    latents = np.random.randn(1,512).astype(np.float32) / 10.0

    r = 1

    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))

        '''
        latents[0, 1] = np.cos(t[png_idx]) * r
        latents[0, 0] = np.sin(t[png_idx]) * r
        latents[0, 2] = np.cos(t[png_idx]) * r
        latents[0, 3] = np.sin(t[png_idx]) * r
        '''
        #latents[0, :32] = np.random.randn(32)

        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%06d.png' % (png_idx)), [0,255], grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()

output = generate_fake_images(num_pngs=1)
