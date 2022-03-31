import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class D_discriminater_Net(nn.Module):
    def __init__(self, num_input_channels):
        super(D_discriminater_Net, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_feature_maps = 128
        self.num_conv_layers = 9
        self.downsampled_channels = 1

        self.intermediate_ecnn = IntermediateECNN(input_features=self.downsampled_channels,
                                                    middle_features=self.num_feature_maps,
                                                    num_conv_layers=self.num_conv_layers)
        self.mlp1 = nn.Linear(4096, 2000)
        self.mlp2 = nn.Linear(2000, 200)
        self.mlp3 = nn.Linear(200, 50)
        self.mlp4 = nn.Linear(50, 1)
        self.mlp5 = nn.Sigmoid()

    def forward(self, x):
        h_dncnn = self.intermediate_ecnn(x)
        x = F.relu(self.mlp1(x.view(h_dncnn.size(0), -1)))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)
        x = self.mlp5(x)
        return x


class IntermediateECNN(nn.Module):
    def __init__(self, input_features, middle_features, num_conv_layers):
        super(IntermediateECNN, self).__init__()
        self.kernel_size = 3
        self.padding = 1
        self.input_features = input_features
        self.num_conv_layers = num_conv_layers
        self.middle_features = middle_features
        self.output_features = 1

        layers = []
        layers.append(nn.Conv2d(in_channels=self.input_features, out_channels=self.middle_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(self.num_conv_layers - 2):
            layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.middle_features,
                                    kernel_size=self.kernel_size, padding=self.padding, bias=False))

            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=self.middle_features, out_channels=self.output_features,
                                kernel_size=self.kernel_size, padding=self.padding, bias=False))
        self.itermediate_encnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.itermediate_encnn(x)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False, bn=False, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class GRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofkernels, nDenselayer, growthRate, numforrg):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GRDB, self).__init__()

        modules = []
        for i in range(numforrg):
            modules.append(RDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate))
        self.rdbs = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(numofkernels * numforrg, numofkernels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = x
        outputlist = []
        for rdb in self.rdbs:
            output = rdb(out)
            outputlist.append(output)
            out = output
        concat = torch.cat(outputlist, 1)
        out = x + self.conv_1x1(concat)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels, nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)

        ###################kingrdb ver2##############################################
        # self.conv_1x1 = nn.Conv2d(nChannels_ + growthRate, nChannels, kernel_size=1, padding=0, bias=False)
        ###################else######################################################
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        # local residual 구조
        out = out + x
        return out

class make_dense(nn.Module):
    def __init__(self, nChannels, nChannels_, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels_, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.nChannels = nChannels

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


def RDB_Blocks(channels, size):
    bundle = []
    for i in range(size):
        bundle.append(RDB(channels, nDenselayer=8, growthRate=64))  # RDB(input channels,
    return nn.Sequential(*bundle)

# Group of group of Residual dense block (GRDB) architecture
class GGRDB(nn.Module):
    """
    https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch
    """

    def __init__(self, numofmodules, numofkernels, nDenselayer, growthRate, numforrg):
        """
        :param nChannels: input feature 의 channel 수
        :param nDenselayer: RDB(residual dense block) 에서 Conv 의 개수
        :param growthRate: Conv 의 output layer 의 수
        """
        super(GGRDB, self).__init__()

        modules = []
        for i in range(numofmodules):
            modules.append(GRDB(numofkernels, nDenselayer=nDenselayer, growthRate=growthRate, numforrg=numforrg))
        self.grdbs = nn.Sequential(*modules)

    def forward(self, x):
        output = x
        for grdb in self.grdbs:
            output = grdb(output)

        return x + output



class ntire_rdb_gd_rir_ver2(nn.Module):
    def __init__(self, input_channel, numofmodules=2, numforrg=4, numofrdb=16, numofconv=8, numoffilters=64, t=1):
        super(ntire_rdb_gd_rir_ver2, self).__init__()

        self.numofmodules = numofmodules # num of modules to make residual
        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t

        self.layer1 = nn.Conv2d(input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        modules = []
        for i in range(self.numofrdb // (self.numofmodules * self.numforrg)):
            modules.append(GGRDB(self.numofmodules, self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg))
        for i in range((self.numofrdb % (self.numofmodules * self.numforrg)) // self.numforrg):
            modules.append(GRDB(self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg))
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(numoffilters, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)
        out = self.cbam(out)

        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x


class G_CBDNet(nn.Module):
    def __init__(self, input_channel, numofmodules=2, numforrg=4, numofrdb=16, numofconv=8, numoffilters=64, t=1):
        super(ntire_rdb_gd_rir_ver2, self).__init__()

        self.numofmodules = numofmodules # num of modules to make residual
        self.numforrg = numforrg  # num of rdb units in one residual group
        self.numofrdb = numofrdb  # num of all rdb units
        self.nDenselayer = numofconv
        self.numofkernels = numoffilters
        self.t = t

        self.layer1 = nn.Conv2d(input_channel, self.numofkernels, kernel_size=3, stride=1, padding=1)
        # self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        modules = []
        for i in range(self.numofrdb // (self.numofmodules * self.numforrg)):
            modules.append(GGRDB(self.numofmodules, self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg))
        for i in range((self.numofrdb % (self.numofmodules * self.numforrg)) // self.numforrg):
            modules.append(GRDB(self.numofkernels, self.nDenselayer, self.numofkernels, self.numforrg))
        self.rglayer = nn.Sequential(*modules)

        self.layer7 = nn.ConvTranspose2d(self.numofkernels, self.numofkernels, kernel_size=4, stride=2, padding=1)

        # self.layer8 = nn.ReLU()
        self.layer9 = nn.Conv2d(self.numofkernels, input_channel, kernel_size=3, stride=1, padding=1)
        self.cbam = CBAM(numoffilters, 16)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = self.layer3(out)

        for grdb in self.rglayer:
            for i in range(self.t):
                out = grdb(out)

        out = self.layer7(out)
        out = self.cbam(out)

        # out = self.layer8(out)
        out = self.layer9(out)

        # global residual 구조
        return out + x