"""Residual nnU-Net with segmentation, centre, affinity and direction heads."""

import importlib
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet


class FiberDetector(nn.Module):
    def __init__(self, plans):
        super().__init__()
        config = plans["configurations"]["3d_fullres"]
        args = dict(config["architecture"]["arch_kwargs"])
        for key in config["architecture"]["_kw_requires_import"]:
            if args[key] is not None:
                mod, name = args[key].rsplit(".", 1)
                args[key] = getattr(importlib.import_module(mod), name)
        self.backbone = ResidualEncoderUNet(
            input_channels=1, num_classes=4, deep_supervision=False, **args
        )
        channels = args["features_per_stage"][0]
        self.centre = nn.Conv3d(channels, 1, 1)
        self.affinity = nn.Conv3d(channels, 6, 1)
        self.direction = nn.Conv3d(channels, 3, 1)
        nn.init.normal_(self.centre.weight, std=0.01)
        nn.init.constant_(self.centre.bias, -2)
        nn.init.normal_(self.affinity.weight, std=0.01)
        nn.init.zeros_(self.affinity.bias)
        nn.init.normal_(self.direction.weight, std=0.01)
        nn.init.zeros_(self.direction.bias)

    def forward(self, x):
        skips = self.backbone.encoder(x)
        decoder = self.backbone.decoder
        x = skips[-1]
        deep = []
        for i, (up, stage) in enumerate(zip(decoder.transpconvs, decoder.stages)):
            x = stage(torch.cat((up(x), skips[-i - 2]), dim=1))
            if i >= len(decoder.stages) - 3:
                deep.append(decoder.seg_layers[i](x))
        return {
            "seg": deep[-1],
            "deep": deep[-2::-1],
            "centre": self.centre(x),
            "affinity": self.affinity(x),
            "direction": self.direction(x),
        }
