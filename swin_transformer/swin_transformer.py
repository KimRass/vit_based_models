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


class SwinTransformer(nn.Module):
    # `patch_size`: $M$
    # `hidden_size`: $C$
    def __init__(self, w, h, n_classes, patch_size=7, hidden_size=96, n_heads=12, dropout_p=0.5):
        patch_size=7
        hidden_size=96

        super().__init__()
        assert w % patch_size == 0 and h % patch_size == 0, "The resolution of the image must be divisible by `patch_size`!"


        n_patches = (w // patch_size) * (h // patch_size)

        patch_partition = Rearrange(pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

        patch_dim = patch_size ** 2 * 3
        linear_embedding = nn.Linear(patch_dim, hidden_size)

        self.swin_transformer_block = SwinTransformerBlock()
        self.patch_merging = PatchMerging()
        stage1 = nn.Sequential(
            [
                linear_embedding(),
                self.swin_transformer_block()
            ]
        )
        stage2 = nn.Sequential(
            [
                self.patch_merging(),
                self.swin_transformer_block()
            ]
        )
    

    def forward(self, image):
        image = torch.randn((4, 3, 175, 280))
        batch_size = image.shape[0]
        w, h = image.shape[2: 4]

        x = patch_partition(image)
        # x = linear_embedding(x)

        for _ in range(2):
            x = stage1(x)
        for _ in range(2):
            x = stage2(x)
        for _ in range(6):
            x = stage3(x)
        for _ in range(2):
            x = stage4(x)
        return x
