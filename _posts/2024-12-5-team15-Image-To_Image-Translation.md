---
layout: post
comments: true
title: Image to Image Translation
author: Tianle Zheng
date: 2024-12-05
---


> This post delves into cutting-edge methods for image-to-image translation and generative modeling, including Pix2Pix, CycleGAN, and FreeControl.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction to Image-to-Image Translation
### What is Image-to-Image Translation?

Image-to-Image Translation is a task in computer vision where the goal is to map an input image from one domain to a corresponding output image in another domain. This process enables diverse applications, such as transforming sketches into photorealistic images, colorizing grayscale photos, or converting summer landscapes into winter scenes.

At its core, Image-to-Image Translation seeks to learn a transformation between two visual domains while preserving the structural content of the input. It has become a cornerstone of many applications in art, medicine, and autonomous systems.

---

### Key Challenges in Translation Tasks

Despite its potential, Image-to-Image Translation faces critical challenges:

1. **Data Requirements**: 
   - Some approaches, such as Pix2Pix, require paired datasets where every input image corresponds to a specific output image. However, collecting such data is often impractical, particularly for artistic tasks or diverse real-world scenarios.

2. **Preserving Structural Consistency**:
   - Maintaining the structural integrity of the input image while transforming it to the target domain is crucial. Failure to do so can lead to outputs that look realistic but misrepresent the input image.

---

In the following sections, we explore how three cutting-edge methods—**Pix2Pix**, **CycleGAN**, and **FreeControl**—tackle these challenges through innovative architectures and loss functions.


## Pix2Pix: Conditional Image-to-Image Translation
### Problem Setting: Paired Image Domains

Pix2Pix tackles the problem of image-to-image translation in scenarios where paired datasets are available. Each input image corresponds to a specific output image, forming a one-to-one mapping. The goal is to learn a transformation that maps pixels from the input domain to the output domain while maintaining structural consistency and generating realistic results.

---

### Core Architecture: U-Net Generator and PatchGAN Discriminator

Pix2Pix uses a U-Net generator and a PatchGAN discriminator in a Conditional GAN framework to generate high-quality and structurally consistent outputs. Unlike unconditional GANs, which generate data samples from random noise without any external constraints, Conditional GANs (cGANs) incorporate additional input, or a condition \( c \), to guide the output generation. In the case of Pix2Pix, the condition is the input image, such as an edge map or a semantic label map, which provides structural guidance to the generator. The discriminator evaluates not only the realism of the generated output but also its alignment with the input condition. This ensures that the generator produces outputs that are both photorealistic and consistent with the structure of the input image, which is crucial for image-to-image translation tasks. The conditional framework allows Pix2Pix to effectively handle paired datasets where each input image has a corresponding output image, enabling precise control over the generation process while maintaining high output quality.

#### 1. Generator: U-Net
The U-Net generator consists of an encoder-decoder architecture with skip connections between layers. The skip connections copy high-resolution features from the encoder to the corresponding decoder layers, preserving spatial details that are often lost in traditional encoder-decoder setups.

#### 2. Discriminator: PatchGAN
The discriminator in Pix2Pix, called PatchGAN, focuses on local texture realism by classifying whether overlapping patches of the output image are real or fake.

- Instead of classifying the entire image, PatchGAN evaluates small \( N X N \) patches (e.g., \( 70 X 70 \)) of the image, ensuring the generator produces realistic details at the patch level.
- This approach reduces the number of parameters in the discriminator and speeds up training while maintaining global coherence.

---

### Objective Functions

Pix2Pix trains its generator and discriminator using two key loss functions: adversarial loss and L1 loss.

#### 1. Adversarial Loss

The adversarial loss encourages the generator to produce outputs indistinguishable from real images. For a conditional GAN, the adversarial loss is:

$$
\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log (1 - D(x, G(x, z)))]
$$

where:
- \( G \): Generator, which takes input \( x \) and outputs \( G(x, z) \) (where \( z \) is optional noise).
- \( D \): Discriminator, which evaluates whether the pair \( (x, y) \) is real or \( (x, G(x, z)) \) is fake.
- \( x \): Input image (e.g., edge map).
- \( y \): Ground truth image (e.g., realistic photo).

During the optimization process, the generator minimizes this loss to fool the discriminator by making \( G(x, z) \) indistinguishable from \( y \), while the discriminator maximizes this loss to correctly classify \( y \) as real and \( G(x, z) \) as fake.

#### 2. L1 Loss

While adversarial loss ensures realism, it doesn’t guarantee that the output matches the input structurally. For instance, \( G(x) \) might produce realistic textures but fail to align with the input \( x \). To address this, Pix2Pix incorporates an \( L1 \) loss:

$$
\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y}[||y - G(x)||_1]
$$

This \( L1 \) loss penalizes the pixel-wise difference between the generated image \( G(x) \) and the ground truth \( y \), and helps the generator produce outputs that are faithful to the input while maintaining realism.

#### 3. Combined Objective

The final objective combines adversarial loss and \( L1 \) loss:

$$
G^* = \arg \min_G \max_D \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)
$$

- \( $$\lambda$$ \): Weighting factor to balance the two losses (e.g., \( $$\lambda = 100$$ \)).
- The adversarial loss ensures realism, while the \( L1 \) loss ensures alignment with the input.

---

### Performance Evaluation

The Pix2Pix model demonstrates outstanding performance across a variety of image-to-image translation tasks. One of the key strengths highlighted in the paper is its ability to generate sharp, high-resolution images with fine-grained details. For example, when translating edge maps into photorealistic images, the model accurately reproduces textures and lighting effects, delivering outputs that are nearly indistinguishable from real photographs. Similarly, in semantic-to-image translation tasks, Pix2Pix preserves the semantic structure provided by the input label map, ensuring that the generated image aligns with the original scene layout.

Quantitative results in the paper show the model outperforms prior approaches in terms of perceptual quality and realism, with the PatchGAN discriminator playing a key role in enforcing local texture realism. Visual comparisons provided in the paper further illustrate its effectiveness, showing that Pix2Pix produces more realistic and coherent images compared to baselines that use simple \( L1 \) or \( L2 \) loss functions alone.

![YOLO]({{ '/assets/images/team17/pix2pix_performance.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


## CycleGAN: Unpaired Image-to-Image Translation

### Problem Setting: Unpaired Image Domains

CycleGAN addresses a critical limitation of paired image-to-image translation methods like Pix2Pix: the requirement for paired datasets. In real-world scenarios, obtaining paired datasets (e.g., a photo and its corresponding Monet-style painting) is impractical. CycleGAN instead enables translation between two domains using unpaired datasets, where images in one domain have no direct correspondence with those in the other.

---

### Core Architecture and Key Innovations

The architecture of CycleGAN builds on the GAN framework but introduces two generators (\( G: X $$\to$$ Y \) and \( F: Y $$\to$$ X \)) and two discriminators (\( D_X \) and \( D_Y \)) to handle unpaired datasets. Several innovations make this possible:

#### 1. Cycle Consistency Loss
The most critical feature of CycleGAN is the cycle consistency loss, which ensures that translations are reversible. As shown in the figure below, if an image \( x \) from domain \( X \) is translated into domain \( Y \) and back to domain \( X \), the reconstructed image \( F(G(x)) \) should match the original \( x \). 

![YOLO]({{ '/assets/images/team17/2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

In unpaired image translation, there is no direct supervision, i.e., no paired images \( (x, y) \) where \( x $$\in$$ X \) and \( y $$\in$$ Y \). Therefore, Cycle Consistency Loss were used to address the descrepancy between the reconstructed image and the original image through the loss function:

$$
\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [||F(G(x)) - x||_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)} [||G(F(y)) - y||_1]
$$

The forward cycle loss corresponds to the first part of the equation and ensures that translating an image \( x \in X \) to \( Y \) using \( G \), and then back to \( X \) using \( F \), reconstructs the original \( x \). The backward cycle loss, represented by the second part of the equation, ensures that translating an image \( y \in Y \) to \( X \) using \( F \), and then back to \( Y \) using \( G \), reconstructs the original \( y \). Both parts use the \( L1 \)-norm to penalize pixel-wise differences, enforcing that the mappings \( G: X \to Y \) and \( F: Y \to X \) are consistent and meaningful. Together, they define the complete cycle consistency loss.


#### 2. Adversarial Loss
CycleGAN uses adversarial losses for both domain translations \( X $$\to$$ Y \) and \( Y $$\to$$ X \). The adversarial loss for \( G \) and \( D_Y \) is defined as:

$$
\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{\text{data}}(y)} [\log D_Y(y)] + \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log (1 - D_Y(G(x)))]
$$

Similarly, a corresponding loss is defined for \( F \) and \( D_X \).

#### 3. Combined Objective
The full objective combines the adversarial and cycle consistency losses, with a hyperparameter \( $$\lambda$$ \) controlling their balance:

$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) + \mathcal{L}_{\text{GAN}}(F, D_X, Y, X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)
$$

#### 4. Network Architecture
CycleGAN adopts a ResNet-based generator architecture with residual blocks for efficient learning of transformations. The generator begins with convolutional layers for feature extraction, followed by residual blocks for transformation, and transposed convolutions for upsampling. The PatchGAN discriminator, which classifies \( N $$\times$$ N \) patches as real or fake, ensures local realism while reducing computational overhead.

---

### Performance Evaluation

CycleGAN demonstrates impressive performance across various unpaired image-to-image translation tasks, showcasing its generality and ability to produce realistic outputs without paired data. The model is evaluated using both qualitative and quantitative metrics, and comparisons against baselines highlight its superiority.

#### 1. Qualitative Results
- **Object Transfiguration**: It performs well on tasks like horse-to-zebra and zebra-to-horse translation, preserving structural details while altering textures and patterns. Similar results are achieved for apple-to-orange translation.
- **Season Transfer**: In seasonal transformations, such as summer-to-winter landscapes, CycleGAN generates visually coherent images that reflect the desired seasonal characteristics.

![YOLO]({{ '/assets/images/team17/3.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}

#### 2. Quantitative Results
CycleGAN’s quantitative performance is evaluated using the following metrics:
- **AMT Perceptual Studies**: On tasks like map↔aerial photo translation, CycleGAN achieved realism scores of 26.8% ± 2.8% and 23.2% ± 3.4% for map-to-photo and photo-to-map translations, respectively. These scores significantly outperform baselines like CoGAN and SimGAN, which barely fool participantsthe Cityscapes labels↔photo task, CycleGAN outperformed most unsupervised methods with a per-pixel accuracy of 52%, per-class accuracy of 17%, and a class IOU of 11%, while remaining slightly below the supervised Pix2Pix (71%, 25%, and 18%)  .

![YOLO]({{ '/assets/images/team17/4.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}


## FreeControl: Training-Free Spatial Control for Text-to-Image Diffusion

### Motivation and Problem Setting

Text-to-Image (T2I) diffusion models have revolutionized high-quality image synthesis, but achieving fine-grained control over the spatial and semantic structure of generated images remains a challenge. Existing methods like ControlNet provide spatial control using guidance images (e.g., edge maps or depth maps), but they require extensive training for each spatial condition, model architecture, and checkpoint. This limits scalability, especially with evolving diffusion models and custom fine-tuned models. Additionally, many training-based methods prioritize spatial conditions at the expense of alignment with the textual description, leading to suboptimal results in cases of conflicting inputs.

FreeControl introduces a training-free framework for controllable T2I generation that works with any spatial condition, model architecture, or checkpoint. It eliminates the need for additional training while balancing spatial alignment with semantic fidelity to the input text, achieving results comparable to training-based methods in quality and control flexibility.

---

### Core Architecture and Key Innovations

#### 1. Semantic Structure Representation
FreeControl leverages the feature space of pre-trained T2I diffusion models to represent and manipulate semantic structures. The key insight is that the principal components of self-attention block features within the U-Net decoder capture spatial and semantic structure across diverse modalities. Using these principal components as a basis, FreeControl ensures spatial alignment with the guidance image without interfering with the image's appearance.

#### 2. Two-Stage Framework
FreeControl operates in two stages as demonstrated in the figure:

![YOLO]({{ '/assets/images/team17/5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

##### (a) Analysis Stage
1. **Seed Images**: Generates a set of seed images using the input text prompt to identify diverse representations of the target concept.
2. **PCA for Semantic Basis**: Performs Principal Component Analysis (PCA) on self-attention features of seed images to derive semantic bases representing the structure.

##### (b) Synthesis Stage
1. **Structure Guidance**: Aligns the generated image's structure with the guidance image by projecting its diffusion features onto the semantic basis with the following loss function:

$$
g_s(\mathbf{s}_t; \mathbf{s}_g, \mathbf{M}) = \frac{\sum_{i,j} m_{ij} \|[\mathbf{s}_t]_{ij} - [\mathbf{s}_g]_{ij}\|_2}{\sum_{i,j} m_{ij}} + w \cdot \frac{\sum_{i,j} (1 - m_{ij}) \| \max([\mathbf{s}_t]_{ij} - \tau_t, 0) \|_2}{\sum_{i,j} (1 - m_{ij})}
$$

This loss equation consists of two terms. The first term enforces structure alignment between the features of the noisy sample (\($$\mathbf{s}_t$$\)) and the guidance features (\($$\mathbf{s}_g$$\)) in regions defined by a binary mask (\($$m_{ij}$$\)). The alignment is achieved by minimizing the \( L2 \)-norm (Euclidean distance) between corresponding features in the masked regions, normalized by the total number of masked locations (\($$\sum_{i,j} m_{ij}$$\)). The second term applies constraint handling in regions where guidance is not enforced (\($$1 - m_{ij}$$\)), penalizing deviations beyond a threshold (\($$\tau_t$$\)) to maintain consistency. The weight \(w\) balances the relative importance of the constraint term compared to the alignment term. Together, these terms ensure precise alignment of structure features with the guidance in specified areas, while preserving stability and quality in the rest of the image.

2. **Appearance Guidance**: Transfers texture and style details from a sibling image (generated without structure guidance) to preserve realistic appearance while adhering to the spatial constraints with the following energy function:

$$
g(\{\mathbf{v}_t(k)\}; \{\bar{\mathbf{v}}_t(k)\}) = \frac{\sum_{k=1}^{N_a} \|\mathbf{v}_t(k) - \bar{\mathbf{v}}_t(k)\|_2^2}{N_a}
$$

By minimizing this energy, FreeControl blends spatial control (enforced by structural guidance) with the photorealistic appearance captured in the sibling image.

3. **Guiding the generation process**: 
The structural guidance and appearance guidance are used as modifiers for the general denoising equation: 

$$
\epsilon_t = (1 + s)\epsilon_\theta(\mathbf{x}_t; t, c) - s\epsilon_\theta(\mathbf{x}_t; t, \emptyset) + \lambda_s g_s + \lambda_a g_a
$$

By combining the baseline prediction with structure and appearance guidance, the updated noise prediction \( \epsilon_t \) incorporates both spatial and aesthetic refinements.

---

### Performance Evaluation

#### Qualitative Results
- **Generalization Across Modalities**: FreeControl demonstrates superior spatial control and semantic alignment across various input conditions, including sketch, depth, edge maps, and segmentation masks. Unlike training-based methods, it effectively handles challenging input conditions such as 2D projections of point clouds or domain-specific shape models (e.g., face or body meshes).
- **Compositional Control**: It supports multiple condition inputs (e.g., combining edge maps with segmentation masks) to generate complex compositions faithful to all inputs.

![YOLO]({{ '/assets/images/team17/6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

#### Quantitative Results
- **CLIP Score**: FreeControl achieves higher image-text alignment compared to training-based baselines (e.g., ControlNet) and training-free methods (e.g., Plug-and-Play).
- **Self-Similarity Distance**: It maintains strong structure preservation, with competitive performance compared to training-based methods.
- **LPIPS**: FreeControl produces images with rich appearance details, avoiding "appearance leakage," a common issue in other training-free approaches.

![YOLO]({{ '/assets/images/team17/7.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

---

## Conclusion

In this blog, we explored key advancements in image-to-image translation and diffusion-based generative models, focusing on Pix2Pix, CycleGAN, and FreeControl. Each method addresses unique challenges in the field—Pix2Pix excels in paired dataset translation, CycleGAN enables unpaired translation through cycle consistency, and FreeControl pioneers training-free spatial control in text-to-image synthesis. By integrating innovations like Conditional GANs, self-supervised losses, and feature-based guidance, these approaches have significantly advanced the ability to generate high-quality, semantically aligned, and spatially controlled images. FreeControl, in particular, demonstrates how powerful generative capabilities can be achieved without retraining, combining structural alignment and photorealistic details in a flexible, scalable framework. Together, these techniques provide a strong foundation for future developments in controllable and interpretable generative models.

## Reference

[1] Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2018.

[2] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." *Proceedings of the IEEE international conference on computer vision*. 2020.

[3] Mo, Sicheng, et al. "FreeControl: Training-Free Spatial Control for Text-to-Image Diffusion." *arXiv preprint arXiv:2312.07536*. 2023.

---