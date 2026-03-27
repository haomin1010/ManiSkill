import torch.nn as nn


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class PlainConv(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_dim=256,
        pool_feature_map=False,
        last_act=True,  # True for ConvBody, False for CNN
    ):
        super().__init__()
        # assume input image size is 128x128

        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        if pool_feature_map:
            # Use AvgPool to a 4x4 spatial grid instead of MaxPool(1,1).
            # This preserves spatial location information (where in the image
            # each feature is), which is critical for object localization tasks.
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = make_mlp(128 * 4 * 4, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(128 * 4 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ResNetEncoder(nn.Module):
    """ResNet18-based visual encoder with spatial feature preservation.

    Adapts the first conv layer to accept multi-channel input (e.g. 4 cameras
    × 3 RGB = 12 channels) by averaging pretrained 3-channel weights across
    groups. Preserves 4×4 spatial features before projecting to out_dim.

    For 128×128 input, ResNet18 produces (512, 4, 4) before global pooling.
    """

    def __init__(self, in_channels=3, out_dim=256, pretrained=True):
        super().__init__()
        import torch
        import torchvision.models as models

        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)

        # Adapt conv1 from 3 channels to in_channels
        old_conv = resnet.conv1  # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained and in_channels != 3:
            # Repeat and average pretrained RGB weights across channel groups.
            # For 12-channel input (4×RGB): copy the 3-channel weights 4 times,
            # then scale down to preserve the expected activation magnitude.
            with torch.no_grad():
                w = old_conv.weight  # (64, 3, 7, 7)
                repeats = in_channels // 3
                remainder = in_channels % 3
                parts = [w] * repeats
                if remainder > 0:
                    parts.append(w[:, :remainder, :, :])
                new_w = torch.cat(parts, dim=1)  # (64, in_channels, 7, 7)
                # Scale down to preserve pre-activation magnitude
                new_w = new_w / (in_channels / 3)
                new_conv.weight.copy_(new_w)
        elif pretrained:
            new_conv.weight.data.copy_(old_conv.weight.data)
        resnet.conv1 = new_conv

        # Build backbone: all layers except global avgpool and fc
        # For 128×128 input: 128→64(conv1)→32(pool)→32(l1)→16(l2)→8(l3)→4(l4)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )  # outputs (B, 512, 4, 4) for 128x128 input

        # Keep 4×4 spatial grid (no-op for 128x128 but safe for other sizes)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(512 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        import torch
        # Apply ImageNet normalization per 3-channel camera group.
        # ResNet18 was pretrained expecting input in range [0,1] normalized
        # with ImageNet mean/std. Without this, pretrained features are useless.
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)
        n_cams = x.shape[1] // 3
        mean_full = mean.repeat(n_cams)[None, :, None, None]  # (1, C, 1, 1)
        std_full  = std.repeat(n_cams)[None, :, None, None]
        x = (x - mean_full) / std_full

        x = self.backbone(x)   # (B, 512, 4, 4)
        x = self.pool(x)       # (B, 512, 4, 4)
        x = x.flatten(1)       # (B, 8192)
        x = self.fc(x)         # (B, out_dim)
        return x
