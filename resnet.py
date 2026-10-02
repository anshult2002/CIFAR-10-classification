"""ResNet-50 (bottleneck, [3, 4, 6, 3]) built from scratch for 32x32 CIFAR input.

CIFAR-style stem: a single 3x3/stride-1 conv and no maxpool (the ImageNet 7x7/stride-2 +
maxpool stem would shrink a 32x32 image to 8x8 before the first residual block).
Kaiming init for convs, zero-init for the last BN in each residual branch.
Regularization: stochastic depth (linear schedule across blocks) + dropout before the FC.
"""
import torch
import torch.nn as nn

NUM_CLASSES = 10


class DropPath(nn.Module):
    """Stochastic depth: zeroes the residual branch per-sample during training and scales
    survivors by 1/keep_prob. No-op at eval time."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # per-sample, broadcast over C,H,W
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.drop_path(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetCifar(nn.Module):
    """Spatial size: 32 -> 32 -> 16 -> 8 -> 4 across the 4 stages.
    drop_path_rate: max stochastic-depth probability (0 at block 1, this value at the last).
    fc_drop_rate: Dropout on the pooled feature vector before the FC head."""

    def __init__(self, block, layers, num_classes=NUM_CLASSES, drop_path_rate=0.1, fc_drop_rate=0.3):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # no maxpool -- keeps full 32x32 resolution entering stage 1
        total_blocks = sum(layers)
        dpr = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        self._dpr_iter = iter(dpr)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)    # 32x32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # 16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   # 8x8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 4x4
        del self._dpr_iter
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=fc_drop_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, channels, num_blocks, stride):
        downsample = None
        out_channels = channels * block.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        blocks = [block(self.in_channels, channels, stride, downsample, drop_path=next(self._dpr_iter))]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(block(self.in_channels, channels, drop_path=next(self._dpr_iter)))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-init the last BN of every residual branch so each block starts as identity.
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_resnet50_cifar(num_classes=NUM_CLASSES, drop_path_rate=0.1, fc_drop_rate=0.3):
    return ResNetCifar(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                       drop_path_rate=drop_path_rate, fc_drop_rate=fc_drop_rate)


if __name__ == "__main__":
    m = build_resnet50_cifar()
    print(f"ResNet-50: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M params")
    print(m(torch.randn(2, 3, 32, 32)).shape)
