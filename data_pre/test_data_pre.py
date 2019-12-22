import os
import shutil

path  = "/mnt/data1/yanghuiyu/datas/imagenet2012/imagenet_val/"
save_path  = "/mnt/data1/yanghuiyu/datas/imagenet2012/val_images/"
txt = "./val.txt"
index = 0
for line in open(txt,"r"):
    name , label = line.strip().split(" ")
    if not os.path.exists(os.path.join(save_path,label)):
        os.mkdir(os.path.join(save_path,label))
    shutil.copy(os.path.join(path,name),os.path.join(save_path,label))
    index +=1
    print(index)