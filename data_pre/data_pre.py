import tarfile
import glob
import os
filelist  = glob.glob("/mnt/data1/yanghuiyu/datas/imagenet2012/imagenet_train/*.tar")
num = 0
print(len(filelist))
# for file in ["n01847000","n02089973","n02091635","n02089078","n02085782","n01749939","n02113978"]:
#     file = os.path.join("/mnt/data1/yanghuiyu/datas/imagenet2012/imagenet_train","{}.tar".format(file))
#     try:
#         folder = file.split("/")[-1].split(".")[0]
#         tar = tarfile.open(file, 'r')
#         tar.extractall( "/mnt/data1/yanghuiyu/datas/imagenet2012/train_imgs/" + folder + "/")  # 可设置解压地址
#         tar.close()
#         num += 1
#     except Exception as e:
#         print(e)
#         pass
#     print("processing %i/1000\r" % num)
#
path  = "/home/yanghuiyu/datas/imagenet/train_imgs"
import  os
dirs  = os.listdir(path)
print(len(dirs))
# for pa in dirs:
#     imgs = os.listdir(os.path.join(path,pa))

    # print(pa,len(imgs))
