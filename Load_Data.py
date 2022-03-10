import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def load_data(path="n_train"):
    try:
        image_list = [f[: -4] for f in os.listdir(f'{path}/TissueImages')]
        data_dir = path
        input_dir = f'{path}/TissueImages'
        target_dir = f'{path}/BinaryMasks'

        X = np.zeros((len(image_list), 444, 444, 3), dtype = np.float32)
        Y = np.zeros((len(image_list), 260, 260), dtype = np.float32)

        for (i, f) in enumerate(tqdm(image_list, desc = 'Loading Data: '.center(34, " "))):
            img = Image.open(os.path.join(input_dir, f + '.png'))
            msk = Image.open(os.path.join(target_dir, f + '.png'))

            X[i] = np.array(img, dtype = np.float32)
            Y[i] = np.array(msk, dtype = np.float32)

        return X, Y
    except Exception as e:
        print("Some Error Occured", e)
        return None, None
