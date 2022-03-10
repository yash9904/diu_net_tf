from Model import DilatedInceptionUNetModel
import argparse
from datetime import datetime
import patch_genarator as patch_genarator
import  mirror_padded_images_generator as mirror_padded_images_generator 
import Load_Data as Load_Data 
import warnings 
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--path', dest='path', type=str, help='path to the dataset folder  like \"MoNuSeg 2018 Training Data\"', default='MoNuSeg 2018 Training Data') 
args = parser.parse_args()
print("")
mirror_padded_images_generator.mirror_pad(args.path)
print("")
patch_genarator.patchImages(mask_root=args.path)
print("")




X, Y = Load_Data.load_data()
print("")
model = DilatedInceptionUNetModel(print_summary=False)
model, history = model.train(X, Y)
model.save(f'/Models_Save/model_{datetime.now().strftime("%H_%M_%S")}.h5')

print("Model Trained And Saved")


