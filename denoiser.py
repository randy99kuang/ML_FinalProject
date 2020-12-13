from imports import *
# UNET from the DL class

# Kernel default = 3
# Stride default = 2
# Pool default = 2

# Functions for adding the convolution layer
def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        # Use batch normalization
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU()
        )


# Upsampling
def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )


# U-Net
class denoiser(nn.Module):
    def __init__(self, up1=4, color=1, useBN=False):
        super(denoiser, self).__init__()
        # Downgrade stages
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)
        self.conv1 = add_conv_stage(color, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)

        # Upgrade stages
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128, 64, useBN=useBN)
        self.conv1m = add_conv_stage(64, 32, useBN=useBN)

        # Maxpool
        self.max_pool = nn.MaxPool2d(2)

        # Upsample layers
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

        # Last layer
        self.convFinal = nn.Conv2d(32, color, kernel_size=1, stride=up1)

        # Activation
        self.activation = nn.Sigmoid()

        # weight initialization
        # You can have your own weight intialization. This is just an example.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.up0(x)
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        # conv4_out = self.conv4(self.max_pool(conv3_out))

        # conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
        # conv3m_out  = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        # Last layer & activation
        final_out = self.convFinal(conv1m_out)

        return (self.activation(final_out))  # Pixels range 0 to 1


class denoiser2(nn.Module):
    def __init__(self, up1=3, color=1):
        super(denoiser2, self).__init__()
        # Downgrade stages
        self.up0 = torch.nn.UpsamplingBilinear2d(scale_factor=up1)
        self.conv1 = nn.Conv2d(color, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.max23 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc = nn.Conv2d(128, 64, 3, padding=1)
        self.up_ed = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.dec = nn.Conv2d(256, color, 3, padding=1)
        self.down = nn.Conv2d(color, color, 3, padding=1, stride=up1)

    def forward(self, x):
        x = self.up0(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max23(x)
        x = F.relu(self.enc(x))
        x = self.up_ed(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dec(x)
        x = self.down(x)

        return x  # Pixels range 0 to 1


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.LR = nn.LeakyReLU(0.1)

        # Encoder
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.BN1 = torch.nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.BN2 = torch.nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.BN3 = torch.nn.BatchNorm2d(32)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(32, 64, 2, stride=2, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)

        # Clean up
        self.final1 = nn.Conv2d(128, 64, 3, padding=1)
        self.final2 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = self.t_conv3(x)
        x = self.final1(x)
        x = self.final2(x)
        x = torch.sigmoid(x)
        return x  # Pixels range 0 to 1



#B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee,
#“Enhanced Deep Residual Networks for Single Image Super-Resolution,”
#arXiv:1707.02921 [cs], Jul. 2017, Accessed: Nov. 29, 2020. [Online].
#Available: https://arxiv.org/abs/1707.02921.

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.5,), rgb_std=(1.0,), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(1).view(1, 1, 1, 1) / std.view(1, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if scale == 1:
            m.append(conv(n_feats, n_feats, 3, bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))

        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, scale=2, conv=default_conv, n_resblocks=2):
        super(EDSR, self).__init__()

        n_feats = 64
        kernel_size = 3
        act = nn.ReLU(True)
        rgb_range = 1
        res_scale = 1
        n_colors = 1
        # self.url = url['r{}f{}x{}'.format(n_resblocks, n_feats, scale)]
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        x = torch.sigmoid(x)
        return x
