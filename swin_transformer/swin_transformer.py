# Reference: https://github.com/berniwal/swin-transformer-pytorch

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class WMSA(nn.Module):
    def __init__(self):
        super().__init__()


class SWMSA(nn.Module):
    def __init__(self):
        super().__init__()


class SwinTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(patch_dim)
        self.wmsa = WMSA()
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP()
        self.swmsa = SWMSA()
    
    def forward(self, image):
        x = self.ln1(image)
        x = self.wmsa(x)
        x += x
        x = self.ln2(x)
        x = self.mlp(x)
        x += x

        x = self.ln1(x)
        x = self.swmsa(x)
        x += x
        x = self.ln2(x)
        x = self.mlp(x)
        x += x
        return x


class PatchPartition(nn.Module):
    def __init__(self, patch_size=4):
        super().__init__()

        self.patch_size = patch_size

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size, padding=0)
    
    def forward(self, input):
        b, c, h, w = input.shape

        new_h = h // self.patch_size
        new_w = w // self.patch_size
        x = self.unfold(input)
        x = x.view((b, -1, new_h, new_w))
        # .permute((0, 2, 3, 1))
        return x


class PatchMerging(nn.Module):
    def __init__(self, downscaling_factor, hidden_size=96):
        super().__init__()

        self.downscaling_factor = downscaling_factor

        self.unfold = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(4 * hidden_size, 2 * hidden_size)
    
    def forward(self, input):
        # input = x
        # input.shape
        b, c, h, w = input.shape

        new_h = h // self.downscaling_factor
        new_w = w // self.downscaling_factor
        x = self.unfold(input)
        x = x.view((b, -1, new_h, new_w))
        x = x.permute((0, 2, 3, 1))
        x = self.linear(x)
        x = x.permute((0, 3, 1, 2))
        return x

    # def __init__(self, downscaling_factor, hidden_size=96):
    #     super().__init__()

    # def forward(self, input):
    #     input = x
    #     downscaling_factor=2
        
    #     b, h, w, c = input.shape

    #     x = input.permute((0, 3, 1, 2))
        
    #     b, c, h, w = x.shape
    #     x.shape
    #     # (4, 96, 43, 70)

    #     new_h = h // downscaling_factor
    #     new_w = w // downscaling_factor
    #     unfold = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
    #     unfold(x).shape
    #     unfold(x).view((b, -1, new_h, new_w)).shape
    #     96*4

        
    #     1680 / hidden_size / 4
    #     1680 / 96
    #     h
    #     new_h
    #     4 * new_h * new_w
        
    #     4*172*1680 / 2940
    #     unfold(input).view((b, new_h, new_w, -1))
    #     shape
        


class SwinTransformer(nn.Module):
    # `window_size`: $M$
    # `hidden_size`: $C$
    def __init__(self, w, h, patch_size=4, n_classes, window_size=7, hidden_size=96, n_layers=(2, 2, 6, 2)):
        # window_size=7
        # patch_size=4
        # hidden_size=96
        downscaling_factors = (4, 2, 2, 2)
        downscaling_factor = downscaling_factors[0]
        
        patch_partition = PatchPartition()
        linear_embedding = nn.Linear(patch_size ** 2 * 3, hidden_size)
        patch_merging = PatchMerging(downscaling_factor=2)

        super().__init__()

        # self.swin_transformer_block = SwinTransformerBlock()
        # stage1 = nn.Sequential(
        #     [
        #         linear_embedding(),
        #         self.swin_transformer_block()
        #     ]
        # )
        # stage2 = nn.Sequential(
        #     [
        #         self.patch_merging(),
        #         self.swin_transformer_block()
        #     ]
        # )
    

    def forward(self, image):
        image = torch.randn((4, 3, 175, 280))
        x = patch_partition(image)
        x = x.permute((0, 2, 3, 1))
        x = linear_embedding(x)
        x = x.permute((0, 3, 1, 2))
        x.shape
        
        x = patch_merging(x)
        x.shape

        for _ in range(2):
            x = stage1(x)
        for _ in range(2):
            x = stage2(x)
        for _ in range(6):
            x = stage3(x)
        for _ in range(2):
            x = stage4(x)
        return x
