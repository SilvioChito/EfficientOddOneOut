import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    # 1x1x1 convolution
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock3d(nn.Module):
    # 3x3x3 Resnet Basic Block
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, drop=0):
        super(BasicBlock3d, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3x3(inplanes, planes, stride, 1, dilation)
        self.bn1 = nn.SyncBatchNorm(planes)
        self.drop1 = nn.Dropout(drop, True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, 1, dilation)
        self.bn2 = nn.SyncBatchNorm(planes)
        self.drop2 = nn.Dropout(drop, True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ProjectedSkipConnection(nn.Module):
    def __init__(self, n):
        super(ProjectedSkipConnection, self).__init__()
        self.conv = conv1x1x1(n, n)
        self.norm = nn.SyncBatchNorm(n)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
        Args:
            x: tensor from encoder
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class EncoderDecoder(nn.Module):
    # 3D network to refine feature volumes

    def __init__(self, channels=[32,64,128], 
                 layers_down=[1,2,3],
                 layers_up=[3,3,3], 
                 norm='BN',
                 drop=0,
                 zero_init_residual=True
                 ):
        
        super(EncoderDecoder, self).__init__()

        self.layers_down = nn.ModuleList()
        self.proj = nn.ModuleList()

        self.layers_down.append(nn.Sequential(*[BasicBlock3d(channels[0], channels[0], drop=drop) for _ in range(layers_down[0]) ]))
        self.proj.append(ProjectedSkipConnection(channels[0]))

        for i in range(1,len(channels)):
            layer = [nn.Conv3d(channels[i-1], channels[i], 3, 2, 1, bias=(norm=='')),
                     nn.SyncBatchNorm(channels[i]),
                     nn.Dropout(drop, True),
                     nn.ReLU(inplace=True)]
            layer += [BasicBlock3d(channels[i], channels[i], drop=drop) for _ in range(layers_down[i])]

            self.layers_down.append(nn.Sequential(*layer))

            if i<len(channels)-1:
                self.proj.append(ProjectedSkipConnection(channels[i]))

        self.proj = self.proj[::-1]

        channels = channels[::-1]

        self.layers_up_conv = nn.ModuleList()
        self.layers_up_res = nn.ModuleList()

        for i in range(1,len(channels)):
            self.layers_up_conv.append(conv1x1x1(channels[i-1], channels[i]))
            self.layers_up_res.append(nn.Sequential( *[BasicBlock3d(channels[i], channels[i], drop=drop) for _ in range(layers_up[i-1]) ]))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock3d):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        xs = []
        for layer in self.layers_down:
            x = layer(x)
            xs.append(x)
        
        xs = xs[::-1]
        out = []

        for i in range(len(self.layers_up_conv)):
            x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
            x = self.layers_up_conv[i](x)

            y = self.proj[i](xs[i+1])
            x = (x + y)/2
            x = self.layers_up_res[i](x)

            out.append(x)

        return out

def build_feat_refinement_block(config):
    return EncoderDecoder(
        config.model.backbone3D.channels, config.model.backbone3D.layers_down,
        config.model.backbone3D.layers, config.model.backbone3D.norm,
        config.model.backbone3D.drop, True
    )
        
def load_decoder(config):
    return build_feat_refinement_block(config)