"""DeepPhys - 2D Convolutional Attention Network.
DeepPhys: Video-Based Physiological Measurement Using Convolutional Attention Networks
ECCV, 2018
Weixuan Chen, Daniel McDuff

Extended with optional TDCM (Temporal Dilated Convolution Module).
"""

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / (xsum + 1e-8) * xshape[2] * xshape[3] * 0.5

    def get_config(self):
        return super(Attention_mask, self).get_config()


class TDCM(nn.Module):
    """
    Temporal Dilated Convolution Module.

    Input:  [B, C, T]
    Output: [B, C, T]
    """

    def __init__(
            self,
            channels=128,
            dropout=0.1,
            kernel_short=3,
            kernel_mid=5,
            kernel_long=7,
            dilation_mid=2,
            dilation_long=3,
    ):
        super(TDCM, self).__init__()

        short_padding = (kernel_short - 1) // 2
        mid_padding = ((kernel_mid - 1) * dilation_mid) // 2
        long_padding = ((kernel_long - 1) * dilation_long) // 2

        self.branch_short = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_short, padding=short_padding, bias=True),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.branch_mid = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_mid,
                padding=mid_padding,
                dilation=dilation_mid,
                bias=True
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.branch_long = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_long,
                padding=long_padding,
                dilation=dilation_long,
                bias=True
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=True),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x

        x_short = self.branch_short(x)
        x_mid = self.branch_mid(x)
        x_long = self.branch_long(x)

        x_cat = torch.cat([x_short, x_mid, x_long], dim=1)
        out = self.fuse(x_cat)

        return out + residual


class DeepPhys(nn.Module):

    def __init__(
            self,
            in_channels=3,
            nb_filters1=32,
            nb_filters2=64,
            kernel_size=3,
            dropout_rate1=0.25,
            dropout_rate2=0.5,
            pool_size=(2, 2),
            nb_dense=128,
            img_size=36,
            use_tdcm=False,
            tdcm_config=None,
    ):
        """Definition of DeepPhys.

        Args:
          in_channels: number of input channels per branch. Default: 3
          img_size: height/width of each frame. Default: 36
          use_tdcm: if True, expects sequence input [B, T, 6, H, W]
          tdcm_config: dict with TDCM hyperparameters
        """
        super(DeepPhys, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.use_tdcm = use_tdcm

        if tdcm_config is None:
            tdcm_config = {}

        # Motion branch convs
        self.motion_conv1 = nn.Conv2d(
            self.in_channels, self.nb_filters1,
            kernel_size=self.kernel_size, padding=(1, 1), bias=True
        )
        self.motion_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1,
            kernel_size=self.kernel_size, bias=True
        )
        self.motion_conv3 = nn.Conv2d(
            self.nb_filters1, self.nb_filters2,
            kernel_size=self.kernel_size, padding=(1, 1), bias=True
        )
        self.motion_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2,
            kernel_size=self.kernel_size, bias=True
        )

        # Appearance branch convs
        self.apperance_conv1 = nn.Conv2d(
            self.in_channels, self.nb_filters1,
            kernel_size=self.kernel_size, padding=(1, 1), bias=True
        )
        self.apperance_conv2 = nn.Conv2d(
            self.nb_filters1, self.nb_filters1,
            kernel_size=self.kernel_size, bias=True
        )
        self.apperance_conv3 = nn.Conv2d(
            self.nb_filters1, self.nb_filters2,
            kernel_size=self.kernel_size, padding=(1, 1), bias=True
        )
        self.apperance_conv4 = nn.Conv2d(
            self.nb_filters2, self.nb_filters2,
            kernel_size=self.kernel_size, bias=True
        )

        # Attention layers
        self.apperance_att_conv1 = nn.Conv2d(self.nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(self.nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        # Avg pooling
        self.avg_pooling_1 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(self.pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(self.pool_size)

        # Dropout layers
        self.dropout_1 = nn.Dropout(self.dropout_rate1)
        self.dropout_2 = nn.Dropout(self.dropout_rate1)
        self.dropout_3 = nn.Dropout(self.dropout_rate1)
        self.dropout_4 = nn.Dropout(self.dropout_rate2)

        # Dense layers
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, self.nb_dense, bias=True)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, self.nb_dense, bias=True)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, self.nb_dense, bias=True)
        else:
            raise Exception('Unsupported image size')

        if self.use_tdcm:
            self.tdcm = TDCM(
                channels=tdcm_config.get("CHANNELS", self.nb_dense),
                dropout=tdcm_config.get("DROPOUT", 0.1),
                kernel_short=tdcm_config.get("KERNEL_SHORT", 3),
                kernel_mid=tdcm_config.get("KERNEL_MID", 5),
                kernel_long=tdcm_config.get("KERNEL_LONG", 7),
                dilation_mid=tdcm_config.get("DILATION_MID", 2),
                dilation_long=tdcm_config.get("DILATION_LONG", 3),
            )
            self.final_dense_2 = nn.Conv1d(self.nb_dense, 1, kernel_size=1, bias=True)
        else:
            self.final_dense_2 = nn.Linear(self.nb_dense, 1, bias=True)

    def _extract_frame_features(self, inputs):
        """
        Frame-wise DeepPhys feature extractor.

        Args:
            inputs: [B, 6, H, W]

        Returns:
            features: [B, nb_dense]
        """
        diff_input = inputs[:, :3, :, :]
        raw_input = inputs[:, 3:, :, :]

        d1 = torch.tanh(self.motion_conv1(diff_input))
        d2 = torch.tanh(self.motion_conv2(d1))

        r1 = torch.tanh(self.apperance_conv1(raw_input))
        r2 = torch.tanh(self.apperance_conv2(r1))

        g1 = torch.sigmoid(self.apperance_att_conv1(r2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        r3 = self.avg_pooling_2(r2)
        r4 = self.dropout_2(r3)

        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(d5))

        r5 = torch.tanh(self.apperance_conv3(r4))
        r6 = torch.tanh(self.apperance_conv4(r5))

        g2 = torch.sigmoid(self.apperance_att_conv2(r6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)

        return d11

    def forward(self, inputs, params=None):
        """
        Baseline mode (use_tdcm=False):
            inputs: [B, 6, H, W]
            output: [B, 1]

        TDCM mode (use_tdcm=True):
            inputs: [B, T, 6, H, W]
            output: [B, T, 1]
        """
        if not self.use_tdcm:
            features = self._extract_frame_features(inputs)
            out = self.final_dense_2(features)
            return out

        # Sequence mode for TDCM
        B, T, C, H, W = inputs.shape
        x = inputs.view(B * T, C, H, W)

        features = self._extract_frame_features(x)      # [B*T, 128]
        features = features.view(B, T, self.nb_dense)   # [B, T, 128]
        features = features.permute(0, 2, 1)            # [B, 128, T]

        features = self.tdcm(features)                  # [B, 128, T]
        out = self.final_dense_2(features)              # [B, 1, T]
        out = out.permute(0, 2, 1)                      # [B, T, 1]

        return out