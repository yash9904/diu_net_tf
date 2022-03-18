import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from PIL import Image

class Augmentation:
    def __init__(self, imdir, msdir):
        self.imlist = [f[: -4] for f in os.listdir(imdir)]
        self.imdir = imdir
        self.msdir = msdir
        self.outdir = 'aug_data'
        self.outdir_im = os.path.join(self.outdir, 'images')
        self.outdir_ms = os.path.join(self.outdir, 'masks')

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        if not os.path.isdir(self.outdir_im):
            os.mkdir(self.outdir_im)
        if not os.path.isdir(self.outdir_ms):
            os.mkdir(self.outdir_ms)

    def save(self, image, file_name, dir):
        cv2.imwrite(os.path.join(dir, file_name), image)

    def flip(self, hor = True, ver = True, fraction = 0):
        for i in tqdm(random.sample(self.imlist, int(fraction * len(self.imlist))), desc = 'Flipping...'):
            im = cv2.imread(os.path.join(self.imdir, i + '.png'))
            ms = cv2.imread(os.path.join(self.msdir, i + '.png'))
            if hor:
                self.save(cv2.flip(im, 1), os.path.join(self.outdir_im, i + 'hor.png'))
                self.save(cv2.flip(ms, 1), os.path.join(self.outdir_ms, i + 'hor.png'))
            if ver:
                self.save(cv2.flip(im, 0), os.path.join(self.outdir_im, i + 'ver.png'))
                self.save(cv2.flip(ms, 0), os.path.join(self.outdir_ms, i + 'ver.png'))

    def ChannelShuffle(self, fraction = 0):
        for i in tqdm(random.sample(self.imlist, int(fraction * len(self.imlist))), desc = 'Shuffling Channels...'):
            im = cv2.imread(os.path.join(self.imdir, i + '.png'))
            ms = cv2.imread(os.path.join(self.msdir, i + '.png'))
            ch = random.choice([0, 1, 2, 3, 4])
            if ch == 0:
                im[:, :, 0], im[:, :, 1], im[:, :, 2] = (im[:, :, 1], im[:, :, 2], im[:, :, 0])
            elif ch == 1:
                im[:, :, 0], im[:, :, 1], im[:, :, 2] = (im[:, :, 2], im[:, :, 0], im[:, :, 1])
            elif ch == 2:
                im[:, :, 0], im[:, :, 1], im[:, :, 2] = (im[:, :, 0], im[:, :, 2], im[:, :, 1])
            elif ch == 3:
                im[:, :, 0], im[:, :, 1], im[:, :, 2] = (im[:, :, 1], im[:, :, 0], im[:, :, 2])
            else:
                im[:, :, 0], im[:, :, 1], im[:, :, 2] = (im[:, :, 2], im[:, :, 1], im[:, :, 0])

            self.save(im, os.path.join(self.outdir_im, i + 'chsh.png'))
            self.save(ms, os.path.join(self.outdir_ms, i + 'chsh.png'))

    def GaussianBlur(self, fraction = 0):
        for i in tqdm(random.sample(self.imlist, int(fraction * len(self.imlist))), desc = 'Applying Gaussian Blur...'):
            im = cv2.imread(os.path.join(self.imdir, i + '.png'))
            ms = cv2.imread(os.path.join(self.msdir, i + '.png'))

            im = cv2.GaussianBlur(im, (5, 5), cv2.BORDER_DEFAULT)

            self.save(im, os.path.join(self.outdir_im, i + 'gb.png'))
            self.save(ms, os.path.join(self.outdir_ms, i + 'gb.png'))
    
    def RandomRotate(self, fraction = 0):
        for i in tqdm(random.sample(self.imlist, int(fraction * len(self.imlist))), desc = 'Applying Gaussian Blur...'):
            im = cv2.imread(os.path.join(self.imdir, i + '.png'))
            ms = cv2.imread(os.path.join(self.msdir, i + '.png'))

            ch = random.choice([0, 1, 2])
            if ch == 0:
                im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ms = cv2.rotate(ms, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif ch == 1:
                im = cv2.rotate(im, cv2.ROTATE_180)
                ms = cv2.rotate(ms, cv2.ROTATE_180)
            else:
                im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                ms = cv2.rotate(ms, cv2.ROTATE_90_CLOCKWISE)

            self.save(im, os.path.join(self.outdir_im, i + 'rr.png'))
            self.save(ms, os.path.join(self.outdir_ms, i + 'rr.png'))
    
def mirror_padding(im_dir, pad_size = 92, im_size = (1000, 1000, 3)):
    '''
        A function to apply mirror padding on tissue images
    '''

    out_dir = os.path.join('monuseg', 'im_train_padded')

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    h, w, d = im_size

    for f in tqdm(os.listdir(im_dir), desc = 'Generating mirror padded images...'):
        blank = np.zeros((h + 2 * pad_size, w + 2 * pad_size, d), dtype = np.uint8)
        im = np.array(Image.open(os.path.join(im_dir, f)))
        blank[pad_size: -pad_size, pad_size: -pad_size] = im
        
        blank[: pad_size, pad_size: -pad_size] = np.flipud(im[: pad_size])
        blank[pad_size: -pad_size, : pad_size] = np.fliplr(im[:, : pad_size])
        blank[-pad_size:, pad_size: -pad_size] = np.flipud(im[-pad_size:])
        blank[pad_size: -pad_size, -pad_size:] = np.fliplr(im[:, -pad_size:])
        
        blank[: pad_size, : pad_size] = np.fliplr(np.flipud(im[: pad_size, : pad_size]))
        blank[: pad_size, -pad_size:] = np.fliplr(np.flipud(im[: pad_size, -pad_size:]))
        blank[-pad_size:, : pad_size] = np.fliplr(np.flipud(im[-pad_size:, : pad_size]))
        blank[-pad_size:, -pad_size:] = np.fliplr(np.flipud(im[-pad_size:, -pad_size:]))
        

        Image.fromarray(blank).save(os.path.join(out_dir, f))
   
def patch(imdir, msdir):
    '''
        A function that patches mirror padded images and masks 
    '''
    out_dir = 'patched_data'
    out_imdir = os.path.join(out_dir, 'tissues')
    out_msdir = os.path.join(out_dir, 'masks')

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if not os.path.isdir(out_imdir):
        os.mkdir(out_imdir)
    if not os.path.isdir(out_msdir):
        os.mkdir(out_msdir)

    img_list = [f[: -4] for f in os.listdir(imdir)]

    for image in tqdm(img_list):
        im = np.array(Image.open(os.path.join(imdir, image + '.tif')), dtype = np.uint8)
        ms = np.array(Image.open(os.path.join(msdir, image + '.png')), dtype = np.uint8)
        
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
                img_patch.save(out_imdir + '/' + image + '_' + str(count) + '.png')
                msk_patch.save(out_msdir + '/' + image + '_' + str(count) + '.png')
                count += 1
