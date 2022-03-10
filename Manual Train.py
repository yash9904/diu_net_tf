import os
import numpy as np
from PIL import Image
from tqdm import tqdm



image_list = [f[: -4] for f in os.listdir('n_train/patched_images')]
data_dir = 'n_train'
input_dir = 'n_train/patched_images'
target_dir = 'n_train/patched_masks'

X = np.zeros((len(image_list), 444, 444, 3), dtype = np.float32)
Y = np.zeros((len(image_list), 260, 260), dtype = np.float32)

for (i, f) in enumerate(tqdm(image_list, desc = 'Loading Data')):
  img = Image.open(os.path.join(input_dir, f + '.png'))
  msk = Image.open(os.path.join(target_dir, f + '.png'))

  X[i] = np.array(img, dtype = np.float32)
  Y[i] = np.array(msk, dtype = np.float32)

print('\nData Loaded!!')


from Model import DilatedInceptionUNetModel
model = DilatedInceptionUNetModel(print_summary=False)
model, history = model.train(X, Y)