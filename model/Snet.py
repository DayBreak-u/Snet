from .modules import *



class SnetExtractor(nn.Module):
    cfg = {
        49: [24, 60, 120, 240, 512],
        146: [24, 132, 264, 528],
        535: [48, 248, 496, 992],
    }

    def __init__(self,  version = 146 , num_classes = 1000 , onnx = False , **kwargs):

        super(SnetExtractor,self).__init__()
        num_layers = [4, 8, 4]
        self.onnx = onnx
        self.num_layers = num_layers
        channels = self.cfg[version]
        self.channels = channels



        self.conv1 = conv_bn(
            3, channels[0], kernel_size=3, stride=2,pad = 1
        )
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1,
        )


        self.stage1 = self._make_layer(
            num_layers[0], channels[0], channels[1], **kwargs)
        self.stage2 = self._make_layer(
            num_layers[1], channels[1], channels[2], **kwargs)
        self.stage3 = self._make_layer(
            num_layers[2], channels[2], channels[3], **kwargs)
        if len(self.channels) == 5:
            self.conv5 = conv_bn(
                channels[3], channels[4], kernel_size=1, stride=1 ,pad=0 )



        if len(channels) == 5:
            self.cem = CEM(channels[-3], channels[-1], channels[-1])
        else:
            self.cem = CEM(channels[-2], channels[-1], channels[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.lastliner1 = nn.Linear(self.channels[-1], 7*7*5)
        # self.relu_last1 = nn.ReLU(inplace=True)
        # self.lastliner2 = nn.Linear(7*7*5, 1024)
        # self.relu_last2 = nn.ReLU(inplace=True)
        self.classifier1 = nn.Linear(self.channels[-1], num_classes)
        self._initialize_weights()

    def _make_layer(self, num_layers, in_channels, out_channels, **kwargs):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(ShuffleV2Block(in_channels, out_channels, mid_channels=out_channels // 2, ksize=5, stride=2))
            else:
                layers.append(ShuffleV2Block(in_channels // 2, out_channels,
                                                    mid_channels=out_channels // 2, ksize=5, stride=1))
            in_channels = out_channels
        return nn.Sequential(*layers)




    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        if len(self.channels) == 5:
            c5 = self.conv5(c5)

        Cglb_lat = self.avgpool(c5)
        Cglb_lat = Cglb_lat.view(Cglb_lat.size(0), -1)

        out  = self.classifier1(Cglb_lat)
        if self.onnx:
            out = F.softmax(out)
        return out



