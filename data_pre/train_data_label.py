import glob
import pandas as pd
import  os
data = pd.read_csv("train.txt",header=None,delimiter=" ")
data["label"] = data[0].map(lambda x:x.split("/")[0])
data = data.drop([0], axis=1)
data = data.drop_duplicates([1,'label'],keep='last')
label_dict  = dict(zip(data["label"],data[1]))
print(label_dict)

train_path = "/mnt/data1/yanghuiyu/datas/imagenet2012/train_imgs"
index = 0
for path in os.listdir(train_path):
    os.replace(os.path.join(train_path,path),os.path.join(train_path,str(label_dict[path])))
    index+=1
    print(index)