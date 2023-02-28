- Reference: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import repeat
from einops.layers.torch import Rearrange

# img = load_image("/Users/jongbeomkim/Downloads/train_images/gt_0.jpg")
# img_tensor = T.ToTensor()(img).unsqueeze(0)
# img_tensor.shape

# class Transformer():
#     def __init__():
#         super().__init__()

#     def forward(self, input):
#         z = nn.LayerNorm(input)


class ViTBase(nn.Module):
    def __init__(self, img_width, img_height, patch_size=16, hidden_size=768, n_heads=12, n_classes):
        super().__init__()
        assert img_width % patch_size == 0 and img_height % patch_size == 0, "The resolution of the image must be divisible by `patch_size`!"

        n_patches = (img_width // patch_size) * (img_height // patch_size)
        
        self.flatten = Rearrange(pattern="b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
        self.linear_projection = nn.Linear(patch_size ** 2 * 3, hidden_size)
        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_size)))
        self.pos_emb = nn.Parameter(torch.randn((1, n_patches + 1, hidden_size)))

        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, img):
        batch_size = img.shape[0]

        z = self.flatten(img)
        z = self.linear_projection(z)
        z = torch.cat(
            tensors=(self.cls_token.repeat(batch_size, 1, 1), z), dim=1
        )
        z += self.pos_emb.repeat(batch_size, 1, 1)

        z = self.transformer(z)
        z = z[:, 0]
        z = self.fc(z)
        return z


if __name__ == "__main__":
    # img_tensor = torch.randn((1, 3, 16 * 11, 16 * 17))
    img_tensor = torch.randn((4, 3, 16 * 11, 16 * 17))

    batch_size, channels, img_height, img_width = img_tensor.shape
    vit_base = ViTBase(img_width=img_width, img_height=img_height)

    vit_base(img_tensor).shape
