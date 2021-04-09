import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import collections


import random


class CSRNet(nn.Module):

    def __init__(self, batch_norm=False, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, batch_norm=False)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True, batch_norm=batch_norm)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = torchvision.models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, label_x):
        label_x = self.frontend(label_x)
        label_x = self.backend(label_x)
        label_x = self.output_layer(label_x)
        label_x = F.interpolate(label_x, scale_factor=8, mode='bilinear', align_corners=True)

        return label_x


def make_layers(layer_list, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for layer in layer_list:
        if layer == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=d_rate, dilation=d_rate)

            if batch_norm:
                layers.extend([conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])

            in_channels = layer

    return nn.Sequential(*layers)


if __name__ == '__main__':
    input_demo = torch.rand((2,3,64,64))
    target_demo = torch.rand((2,1,64,64))

    model = CSRNet()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-5,momentum=0.95,weight_decay=5e-4)

    output_demo = model(input_demo)
    print(output_demo.shape)

    # model.train()
    # for i in range(10):
    #     output = model(input_demo)
    #     loss = criterion(output_demo, target_demo)
    #     print('epoch{}: '.format(i), loss.item())
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()