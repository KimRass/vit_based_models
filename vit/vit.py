- Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops.layers.torch import Rearrange

# image = load_image("/Users/jongbeomkim/Downloads/train_images/gt_0.jpg")
# image = T.ToTensor()(image).unsqueeze(0)
# image.shape

# class Transformer():
#     def __init__():
#         super().__init__()

#     def forward(self, input):
#         z = nn.LayerNorm(input)


class ViTBase(nn.Module):
    def __init__(self, width, height, n_classes, patch_size=16, hidden_size=768, n_heads=12, dropout_p=0.5):
        super().__init__()
        assert width % patch_size == 0 and height % patch_size == 0, "The resolution of the image must be divisible by `patch_size`!"

        n_patches = (width // patch_size) * (height // patch_size)
        

        self.flatten = Rearrange(pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

        patch_dim = patch_size ** 2 * 3
        self.layernorm1 = nn.LayerNorm(patch_dim)
        self.linear_projection = nn.Linear(patch_dim, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)

        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_size)))
        self.pos_embed = nn.Parameter(torch.randn((1, n_patches + 1, hidden_size)))
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, image):
        batch_size = image.shape[0]

        x = self.flatten(image)
        x = self.layernorm1(x) # Not in original paper
        x = self.linear_projection(x)
        x = self.layernorm2(x) # Not in original paper
        x = torch.cat(
            tensors=(self.cls_token.repeat(batch_size, 1, 1), x), dim=1
        )
        x += self.pos_embed.repeat(batch_size, 1, 1)
        x = self.dropout(x) # Not in original paper

        x = self.transformer(x)
        x = x[:, 0]
        x = self.layernorm2(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    image = torch.randn((4, 3, 16 * 11, 16 * 17))

    batch_size, channels, height, width = image.shape
    vit_base = ViTBase(width=width, height=height, n_classes=1_000)

    vit_base(image).shape
