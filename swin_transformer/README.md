# Paper Summary
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
- ***Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. We observe that significant challenges in transferring its high performance in the language domain to the visual domain can be explained by differences between the two modalities. One of these differences involves scale. Unlike the word tokens that serve as the basic elements of processing in language Transformers, visual elements can vary substantially in scale, a problem that receives attention in tasks such as object detection. In existing Transformer-based models [64, 20], tokens are all of a fixed scale, a property unsuitable for these vision applications.***
- ***Another difference is the much higher resolution of pixels in images compared to words in passages of text. There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size.***
- To address these differences, we propose a hierarchical Transformer whose representation is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and ***has linear computational complexity with respect to image size.***
- ***It is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared.***
## Related Works
### CNN and variants
- There has been much work on improving individual convolution layers, such as depthwise convolution [70] and deformable convolution [18] [84].
### Transformer based vision backbones
- Most related to our work is the Vision Transformer (ViT) [20] and its follow-ups [63] [72] [15] [28] [66]. While ViT requires large-scale training datasets (i.e., JFT-300M) to perform well, ***DeiT [63] introduces several training strategies that allow ViT to also be effective using the smaller ImageNet-1K dataset. The results of ViT on image classification are encouraging, but its architecture is unsuitable for use as a general-purpose backbone network on dense vision tasks or when the input image resolution is high, due to its low-resolution feature maps and the quadratic increase in complexity with image size.***
## Methodology
- Figure 1. Hierarchical feature maps
    - <img src="https://i.imgur.com/zrQ46ny.png" width="500">
- To overcome these issues, we propose a general-purpose Transformer backbone, called Swin Transformer, ***which constructs hierarchical feature maps and has linear computational complexity to image size. Swin Transformer constructs a hierarchical representation by starting from small-sized patches (outlined in gray) and gradually merging neighboring patches in deeper Transformer layers.*** With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as feature pyramid networks (FPN) [42] or U-Net [51]. ***The linear computational complexity is achieved by computing self-attention locally within non-overlapping windows that partition an image (outlined in red). The number of patches in each window is fixed, and thus the complexity becomes linear to image size.***
- ***The proposed Swin Transformer builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red).*** It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks.
- In contrast, previous vision Transformers [20] produce feature maps of a single low resolution and ***have quadratic computation complexity to input image size due to computation of self-attention globally.***
- Figure 2. Shifted window approach
    - <img src="https://i.imgur.com/gARWp3b.png" width="500">
- ***A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers. The shifted windows bridge the windows of the preceding layer, providing connections among them*** that significantly enhance modeling power.
- In layer $l$ (left), a regular window partitioning scheme is adopted, and self-attention is computed within each window. In the next layer $l + 1$ (right), the window partitioning is shifted, resulting in new windows. The self-attention computation in the new windows crosses the boundaries of the previous windows in layer $l$, providing connections among them.
## Architecture
- Figure 3. SwinT architecture
    - <img src="https://i.imgur.com/CbHT8DG.png" width="800">
- It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is treated as a "token" and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of $4 \times 4$ and thus the feature dimension of each patch is $4 \times 4 \times 3 = 48$. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as $C$). Several ***Transformer blocks with modified self-attention computation (Swin Transformer blocks)*** are applied on these patch tokens. The Transformer blocks maintain the number of tokens ($\frac{H}{4} × \frac{W}{4}$), and together with the linear embedding are referred to as "Stage 1". To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of $2 \times 2$ neighboring patches, and applies a linear layer on the $4C$-dimensional concatenated features. This reduces the number of tokens by a multiple of $2 \times 2 = 4$ (2× downsampling of resolution), and the output dimension is set to $2C$. Swin Transformer blocks are applied afterwards for feature transformation, with the resolution kept at $\frac{H}{8} × \frac{W}{8}$. This first block of patch merging and feature transformation is denoted as "Stage 2". The procedure is repeated twice, as "Stage 3" and "Stage 4", with output resolutions of $\frac{H}{16} × \frac{W}{16}$ and $\frac{H}{32} × \frac{W}{32}$, respectively. These stages jointly produce a hierarchical representation, with the same feature map resolutions as those of typical convolutional networks, e.g., VGG and ResNet. As a result, the proposed architecture can conveniently replace the backbone networks in existing methods for various vision tasks.
## References
- [20] [An image is worth 16x16 words: Transformers for image recognition at scale]
- [63] [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)
- [64] [Attention is all you need]
- [70] [Aggregated residual transformations for deep neural networks]