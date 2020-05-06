import torch
import torch.nn as nn

# VGG的结构顺序, 描述了各层的输出通道数/最大池化层
# structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# structure = [32, 'M', 96, 'M', 148, 148, 'M', 256, 'M', 256, 'M']
structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGNet(nn.Module):
    def __init__(self, se_block, n_classes=10):
        '''

        :param se_block: 是否使用se_block
        :param n_classes: 类别数
        '''
        super(VGGNet, self).__init__()
        self.conv_layers = None
        self.conv_out_channels = self.make_conv_layers(se_block)   # 构建卷积层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层
        self.fc_layers = nn.Sequential(  # 三层全连接层
            # 全连接层1
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1 * 1 * self.conv_out_channels, out_features=n_classes),
            # nn.ReLU(inplace=True),
            # # 全连接层2
            # nn.Dropout(p=0.5),
            # nn.Linear(in_features=1024, out_features=1024),
            # nn.ReLU(inplace=True),
            # 全连接层3
            # nn.Linear(in_features=512, out_features=n_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def make_conv_layers(self, se_block):
        '''
        构造所有卷积层
        :param se_block: 是否使用se_block
        '''
        layers = []
        in_channels = 3
        for v in structure:
            if v == 'M':  # max_pooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if se_block:
                    se = SeBlock(in_channels, v, kernel_size=3, stride=1, padding=1)
                    layers += [se]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.conv_layers = nn.Sequential(*layers)
        return in_channels


class SeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, r=16):
        super(SeBlock, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        # 正常的卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            )
        # Squeeze
        self.sq = nn.AdaptiveAvgPool2d((1, 1))
        # Excitation
        self.ex = nn.Sequential(
            nn.Linear(in_features=out_channels, out_features=int(out_channels / r)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=int(out_channels / r), out_features=out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        u = self.conv(x)
        sq = self.sq(u)
        sq = sq.view(-1, self.out_channels)
        ex = self.ex(sq)
        ex = ex.view(-1, self.out_channels, 1, 1)
        return ex * u
