import torch.nn as nn
import torch

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        # Compute attention weights
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # Apply sigmoid to get the attention map
        attention_map = weights.sigmoid()

        # Use the attention map to modulate the features
        output = (group_x * attention_map).reshape(b, c, h, w)

        # Return both the modulated output and the attention map
        return output, attention_map


##This is multiscale residual block code from: https://github.com/MIVRC/MSRN-PyTorch/blob/master/MSRN/Train/model/common.py


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MSRB(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class finalMSRB(nn.Module):
    def __init__(self, conv=default_conv, n_feats=64):
        super(finalMSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        output = self.sigmoid(output)
        return output


# this code incorporates multiscale residuals
class newAECBAM3(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(newAECBAM3, self).__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.res1 = MSRB(n_feats=64)
        self.se1 = EMA(64)  # CBAM2(64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.res2 = MSRB(n_feats=128)
        self.se2 = EMA(128)  # CBAM2(128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.res3 = MSRB(n_feats=256)
        self.se3 = EMA(256)  # CBAM2(256)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.res4 = MSRB(n_feats=512)
        self.se4 = EMA(512)  # (512)

        self.embedding = nn.Linear(512 * 14 * 14, latent_dim)

        # Decoder
        self.deconv1 = nn.Sequential(nn.Linear(latent_dim, 512 * 14 * 14))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.res5 = MSRB(n_feats=256)
        self.se5 = EMA(256)  # CBAM2(256)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.res6 = MSRB(n_feats=128)
        self.se6 = EMA(128)  # CBAM2(128)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.res7 = MSRB(n_feats=64)
        self.se7 = EMA(64)  # CBAM2(64)

        # self.deconv5 = nn.Sequential(
        #    nn.ConvTranspose2d(64, input_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
        #    nn.Sigmoid()
        # )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, input_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(input_dim[0]),
            nn.LeakyReLU(inplace=True)
        )
        self.res8 = finalMSRB(n_feats=input_dim[0])

    def encode(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x, attn = self.se1(x)

        x = self.conv2(x)
        x = self.res2(x)
        x, attn = self.se2(x)

        x = self.conv3(x)
        x = self.res3(x)
        x, attn = self.se3(x)

        x = self.conv4(x)
        x = self.res4(x)
        x, attn = self.se4(x)
        # Flatten and pass through the linear layer to get the latent vector
        latent = self.embedding(x.view(x.shape[0], -1))
        return latent

    def forward(self, x, return_attn=False):
        attn_maps = {}
        x = self.conv1(x)
        x = self.res1(x)
        x, attn = self.se1(x)
        if return_attn:
            attn_maps['se1'] = attn
        # Continue with other layers (you may also collect from se2, se3, etc.)
        x = self.conv2(x)
        x = self.res2(x)
        x, attn = self.se2(x)
        if return_attn:
            attn_maps['se2'] = attn
        x = self.conv3(x)
        x = self.res3(x)
        x, attn = self.se3(x)
        if return_attn:
            attn_maps['se3'] = attn
        x = self.conv4(x)
        x = self.res4(x)
        x, attn = self.se4(x)
        if return_attn:
            attn_maps['se4'] = attn

        # ... Continue with the decoder
        x = self.embedding(x.view(x.shape[0], -1))
        x = self.deconv1(x)
        x = x.view(-1, 512, 14, 14)
        x = self.deconv2(x)
        x = self.res5(x)
        x, attn = self.se5(x)
        if return_attn:
            attn_maps['se5'] = attn
        x = self.deconv3(x)
        x = self.res6(x)
        x, attn = self.se6(x)
        if return_attn:
            attn_maps['se6'] = attn
        x = self.deconv4(x)
        x = self.res7(x)
        x, attn = self.se7(x)
        if return_attn:
            attn_maps['se7'] = attn
        x = self.deconv5(x)
        x = self.res8(x)

        if return_attn:
            return x, attn_maps
        else:
            return x


## just an extra layer
class newAECBAM4(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(newAECBAM4, self).__init__()
        # ---------------------------
        # Encoder
        # ---------------------------
        # Extra block: maps input channels to 32 channels with stride 2 (downsampling)
        self.extra_conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.extra_res = MSRB(n_feats=32)
        self.extra_se = EMA(
            32)  # if you want to capture this attention map wil need to change the train script to do so

        # conv1: takes 32 channels to 64 channels with stride 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.res1 = MSRB(n_feats=64)
        self.se1 = EMA(64)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.res2 = MSRB(n_feats=128)
        self.se2 = EMA(128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.res3 = MSRB(n_feats=256)
        self.se3 = EMA(256)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.res4 = MSRB(n_feats=512)
        self.se4 = EMA(512)

        # For an input of 224x224:
        # extra_conv: 224 -> 112, conv1: 112 -> 56, conv2: 56 -> 28, conv3: 28 -> 14, conv4: 14 -> 7
        self.embedding = nn.Linear(512 * 7 * 7, latent_dim)

        # ---------------------------
        # Decoder
        # ---------------------------
        self.deconv1 = nn.Sequential(nn.Linear(latent_dim, 512 * 7 * 7))

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.res5 = MSRB(n_feats=256)
        self.se5 = EMA(256)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.res6 = MSRB(n_feats=128)
        self.se6 = EMA(128)

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.res7 = MSRB(n_feats=64)
        self.se7 = EMA(64)

        # Extra decoder block: mirror of the encoder's extra block
        self.extra_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.extra_res_decoder = MSRB(n_feats=32)
        self.extra_se_decoder = EMA(32)

        # Final deconvolution: from 32 channels to the original input channels,
        # upsampling from 112 -> 224
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, input_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(input_dim[0]),
            nn.LeakyReLU(inplace=True)
        )
        self.res_final = finalMSRB(n_feats=input_dim[0])

    def encode(self, x):
        # Encoder: extra block first
        x = self.extra_conv(x)  # [input_dim[0] -> 32] with downsampling: 224 -> 112
        x = self.extra_res(x)
        x, attn_extra = self.extra_se(x)

        x = self.conv1(x)  # 112 -> 56, 32 -> 64
        x = self.res1(x)
        x, attn1 = self.se1(x)

        x = self.conv2(x)  # 56 -> 28, 64 -> 128
        x = self.res2(x)
        x, attn2 = self.se2(x)

        x = self.conv3(x)  # 28 -> 14, 128 -> 256
        x = self.res3(x)
        x, attn3 = self.se3(x)

        x = self.conv4(x)  # 14 -> 7, 256 -> 512
        x = self.res4(x)
        x, attn4 = self.se4(x)

        # Flatten and embed
        latent = self.embedding(x.view(x.shape[0], -1))
        return latent

    def decode(self, x, return_attn=False):
        x = self.deconv1(x)
        x = x.view(-1, 512, 7, 7)

        x = self.deconv2(x)
        x = self.res5(x)
        x, attn = self.se5(x)
        if return_attn:
            attn_maps['se5'] = attn

        x = self.deconv3(x)
        x = self.res6(x)
        x, attn = self.se6(x)
        if return_attn:
            attn_maps['se6'] = attn

        x = self.deconv4(x)
        x = self.res7(x)
        x, attn = self.se7(x)
        if return_attn:
            attn_maps['se7'] = attn

        # Extra decoder block (mirrors the encoder's extra block)
        x = self.extra_deconv(x)  # Upsample from 56 -> 112, 64 -> 32
        x = self.extra_res_decoder(x)
        x, attn = self.extra_se_decoder(x)
        if return_attn:
            attn_maps['se8'] = attn

        # Final deconvolution to return to original input dimensions (112 -> 224)
        x = self.final_deconv(x)
        x = self.res_final(x)

        if return_attn:
            return x, attn_maps
        else:
            return x

    def forward(self, x, return_attn=False):
        attn_maps = {}
        # Encoder: extra block first
        x = self.extra_conv(x)
        x = self.extra_res(x)
        x, attn = self.extra_se(x)
        if return_attn:
            attn_maps['se0'] = attn

        x = self.conv1(x)
        x = self.res1(x)
        x, attn = self.se1(x)
        if return_attn:
            attn_maps['se1'] = attn

        x = self.conv2(x)
        x = self.res2(x)
        x, attn = self.se2(x)
        if return_attn:
            attn_maps['se2'] = attn

        x = self.conv3(x)
        x = self.res3(x)
        x, attn = self.se3(x)
        if return_attn:
            attn_maps['se3'] = attn

        x = self.conv4(x)
        x = self.res4(x)
        x, attn = self.se4(x)
        if return_attn:
            attn_maps['se4'] = attn
        x = self.embedding(x.view(x.shape[0], -1))

        # Decoder: reshape from latent vector

        x = self.deconv1(x)
        x = x.view(-1, 512, 7, 7)

        x = self.deconv2(x)
        x = self.res5(x)
        x, attn = self.se5(x)
        if return_attn:
            attn_maps['se5'] = attn

        x = self.deconv3(x)
        x = self.res6(x)
        x, attn = self.se6(x)
        if return_attn:
            attn_maps['se6'] = attn

        x = self.deconv4(x)
        x = self.res7(x)
        x, attn = self.se7(x)
        if return_attn:
            attn_maps['se7'] = attn

        # Extra decoder block (mirrors the encoder's extra block)
        x = self.extra_deconv(x)  # Upsample from 56 -> 112, 64 -> 32
        x = self.extra_res_decoder(x)
        x, attn = self.extra_se_decoder(x)
        if return_attn:
            attn_maps['se8'] = attn

        # Final deconvolution to return to original input dimensions (112 -> 224)
        x = self.final_deconv(x)
        x = self.res_final(x)

        if return_attn:
            return x, attn_maps
        else:
            return x








