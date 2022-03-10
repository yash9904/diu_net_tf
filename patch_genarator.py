import os
from os.path import isdir, join
from os import mkdir, listdir
import numpy as np
from PIL import Image
from tqdm import tqdm
import os


    
# img_root = 'mirror_padded_ims'
# mask_root = 'MoNuSeg 2018 Training Data/Binary_masks'



def patchImages(img_root="mirror_padded_ims", mask_root="MoNuSeg 2018 Training Data"):
    try:
        mask_root = f'{mask_root}/Binary_masks'
        patched_img__dir = 'n_train/TissueImages'
        patched_masks_dir = 'n_train/BinaryMasks'

        img_list = [f[: -4] for f in listdir(img_root)]


        if not isdir('n_train'):
            mkdir('n_train')
        if not isdir('n_train/TissueImages'):
            mkdir('n_train/TissueImages')
        if not isdir('n_train/BinaryMasks'):
            mkdir('n_train/BinaryMasks')

        for image in tqdm(img_list, desc="Patching Images: ".center(34, " ")):
            im = np.array(Image.open(join(img_root, image + '.png')), dtype = np.uint8)
            ms = np.array(Image.open(join(mask_root, image + '.png')), dtype = np.uint8)
            
            count = 0
            for i in range(0, 781, 260):
                for j in range(0, 781, 260):
                    if i >= 780 or j >= 780:
                        if i >= 780:
                            i = 740
                        if j >= 780:
                            j = 740
                    img_patch = Image.fromarray(im[0 + i: 444 + i, 0 + j: 444 + j])
                    msk_patch = Image.fromarray(ms[0 + i: 260 + i, 0 + j: 260 + j])
                    img_patch.save(patched_img__dir + '/' + image + '_' + str(count) + '.png')
                    msk_patch.save(patched_masks_dir + '/' + image + '_' + str(count) + '.png')
                    count += 1
        return True
    except Exception as e:
        print('Error in patching images', e)
        return False

