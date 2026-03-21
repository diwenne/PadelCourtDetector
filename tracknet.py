"""
TrackNet-based encoder-decoder CNN for heatmap regression.

Architecture:
    Encoder:  3×Conv64 → Pool → 2×Conv128 → Pool → 3×Conv256 → Pool → 3×Conv512
    Decoder:  Upsample → 3×Conv256 → Upsample → 2×Conv128 → Upsample → 2×Conv64 → Conv(out_channels)

Each ConvBlock = Conv2d(3×3) → ReLU → BatchNorm2d.

The network operates at half the input resolution due to 3 pooling layers (8× downsample)
followed by 3 upsampling layers (8× upsample), but input is already pre-halved by the
dataset loader, so effective output = input_size / 2.

Input:  (B, 3, H, W)  — RGB image normalized to [0, 1], CHW layout
Output: (B, out_channels, H, W) — raw logits (apply sigmoid for heatmap probabilities)

Originally from: https://github.com/yastrebksv/TennisCourtDetector
Modified for padel court keypoint detection with configurable out_channels.
"""
import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """Conv2d → ReLU → BatchNorm2d building block.
    
    Args:
        in_channels:  Number of input feature channels.
        out_channels: Number of output feature channels.
        kernel_size:  Convolution kernel size (default: 3).
        pad:          Padding (default: 1, preserves spatial dimensions with 3×3 kernel).
        stride:       Convolution stride (default: 1).
        bias:         Whether to include bias in Conv2d (default: True).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    """Fully-convolutional encoder-decoder for heatmap regression.
    
    Predicts one heatmap per output channel. For padel court detection:
        - out_channels=6: [tol, tor, point_7, point_9, tom, bottom_t]
    
    Each heatmap is a 2D Gaussian centered on the keypoint location.
    The network outputs raw logits; apply sigmoid to get probabilities.
    
    Args:
        out_channels: Number of output heatmap channels (default: 14 for tennis,
                      use 6 for padel court detection).
    
    Input shape:  (B, 3, H, W)
    Output shape: (B, out_channels, H, W)  — same spatial dimensions as input
    
    Weight initialization:
        - Conv2d weights: Uniform(-0.05, 0.05)
        - Conv2d biases: Zero
        - BatchNorm weights: 1, biases: 0
    """
    def __init__(self, out_channels=14):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self._init_weights()
                  
    def forward(self, x):
        """Forward pass through encoder-decoder.
        
        Args:
            x: Input tensor of shape (B, 3, H, W), normalized to [0, 1].
        
        Returns:
            Raw logit heatmaps of shape (B, out_channels, H, W).
        """
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x
    
    def _init_weights(self):
        """Initialize weights using uniform distribution for Conv2d and
        standard initialization for BatchNorm2d layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)   
                
if __name__ == '__main__':
    device = 'cpu'
    model = BallTrackerNet().to(device)
    inp = torch.rand(1, 3, 360, 640)
    out = model(inp)
    print('out = {}'.format(out.shape))
    
    