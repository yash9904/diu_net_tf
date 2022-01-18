from os.path import isdir, join
from os import mkdir, listdir
import numpy as np
from PIL import Image

if not isdir('train'):
    mkdir('train')
if not isdir('train\patched_images'):
    mkdir('train\patched_images')
if not isdir('train\patched_masks'):
    mkdir('train\patched_masks')

    

img_root = 'MoNuSeg 2018 Training Data\Tissue Images'
mask_root = 'MoNuSeg 2018 Training Data\Binary_masks'

patched_img__dir = 'train\patched_images'
patched_masks_dir = 'train\patched_masks'

img_list = [f[: -4] for f in listdir(img_root)]

print('Creating train patches ...')

for image in img_list:
    im = np.array(Image.open(join(img_root, image + '.tif')).resize((1024, 1024)), dtype = np.uint8)
    ms = np.array(Image.open(join(mask_root, image + '.png')).resize((1024, 1024)), dtype = np.uint8)
    
    print(f'Creating patches of image and masks for {image}')

    count = 0
    for i in range(0, 769, 256):
        for j in range(0, 769, 256):
            img_patch = Image.fromarray(im[0 + i: 256 + i, 0 + j: 256 + j])
            msk_patch = Image.fromarray(ms[0 + i: 256 + i, 0 + j: 256 + j])
            img_patch.save(patched_img__dir + '\\' + image + '_' + str(count) + '.png')
            msk_patch.save(patched_masks_dir + '\\' + image + '_' + str(count) + '.png')
            count += 1
    print(f'Patches successfully generated for {image}')

print('Train patches successfully generated!')
