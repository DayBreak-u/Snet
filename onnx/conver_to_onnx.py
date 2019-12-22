import  sys
sys.path.insert(0,"../model")
from  torch import  nn
import torch
from Snet import  SnetExtractor
from utils import  load_model


net  = SnetExtractor(onnx=True)

net = load_model(net, "../checkpoints/model_best.pth.tar")
net.eval()
print('Finished loading model!')
print(net)
device = torch.device("cpu")
net = net.to(device)

##################export###############
output_onnx = 'snet146.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input"]
# output_names = ["hm" , "wh"  , "reg"]
output_names = ["cls_prob"  ]
inputs = torch.randn(1, 3, 224, 224).to(device)
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)