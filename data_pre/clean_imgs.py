import shutil
from PIL import  Image
import  glob
import os
from tqdm import  tqdm
path  = "/home/yanghuiyu/datas/imagenet/val_images/*/*"
imgs = glob.glob(path)
for path in tqdm(imgs):
    try:
        Image.open(path)
    except:
        print(path)
        os.remove(path)