import os
from tqdm import tqdm
import numpy as np
from PIL import Image

def mirror_pad(im_dir = "MoNuSeg 2018 Training Data", out_dir="mirror_padded_ims"):
    try:
        im_dir = im_dir + "/Tissue_Images"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)


        for f in tqdm(os.listdir(im_dir), desc = 'Generating mirror padded images...'):
            blank = np.zeros((1184, 1184, 3), dtype = np.uint8)
            im = np.array(Image.open(os.path.join(im_dir, f)))
            blank[92: 1092, 92: 1092] = im
            
            blank[: 92, 92: 1092] = np.flipud(im[: 92])
            blank[92: 1092, : 92] = np.fliplr(im[:, : 92])
            blank[-92:, 92: 1092] = np.flipud(im[-92:])
            blank[92: 1092, -92:] = np.fliplr(im[:, -92:])
            
            blank[: 92, : 92] = np.fliplr(np.flipud(im[: 92, : 92]))
            blank[: 92, -92:] = np.fliplr(np.flipud(im[: 92, -92:]))
            blank[-92:, : 92] = np.fliplr(np.flipud(im[-92:, : 92]))
            blank[-92:, -92:] = np.fliplr(np.flipud(im[-92:, -92:]))
            
            Image.fromarray(blank).save(os.path.join(out_dir, f))
    except Exception as e:
        print("Some Error Occured: `{}`".format(e))

