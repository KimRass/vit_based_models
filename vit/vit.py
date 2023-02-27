- Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import repeat
from einops.layers.torch import Rearrange

img = load_image("/Users/jongbeomkim/Downloads/train_images/gt_0.jpg")
img_tensor = T.ToTensor()(img).unsqueeze(0)
img_tensor.shape


class ViTBase(nn.Module):
    def __init__(self, img_width, img_height, patch_size=16, hidden_size=768, n_heads=12):
        super().__init__()
        assert img_width % patch_size == 0 and img_height % patch_size == 0, "The resolution of the image must be divisible by `patch_size`!"
        n_patches = (img_width // patch_size) * (img_height // patch_size)
        
        self.flatten = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        self.linear_projection = nn.Linear(patch_size ** 2 * 3, hidden_size)

        # z = self.flatten(img_tensor)
        # z = self.linear_projection(z)
        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_size)))
        # z = torch.cat((self.cls_token, z), dim=1)
        self.pos_emb = nn.Parameter(torch.randn((1, n_patches + 1, hidden_size)))

    def forward(self, img):
        z = self.flatten(img)
        z = self.linear_projection(z)
        z = torch.cat((self.cls_token, z), dim=1)
        z += self.pos_emb
        return z


if __name__ == "__main__":
    img_tensor = torch.randn((4, 3, 16 * 11, 16 * 17))

    batch_size, channels, img_height, img_width = img_tensor.shape
    vit_base = ViTBase(img_width=img_width, img_height=img_height)
    vit_base(img_tensor).shape


    # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)