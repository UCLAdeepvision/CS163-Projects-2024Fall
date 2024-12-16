---
layout: post
comments: true
title: Super-resolution
author: Nicholas Chu, James Feeney, Tyler Nguyen, Jonny Xu
date: 2024-01-01
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

![Example of Low-Resolution vs. High-Resolution Images](/assets/images/UCLAdeepvision/team17/SR.png)
*Figure 1: An illustration of SR enhancing image clarity.*

Super-resolution (SR) is a transformative task in computer vision, aimed at enhancing the spatial resolution of images or videos by reconstructing high-resolution (HR) content from low-resolution (LR) inputs. The problem of SR is not only about visual appeal; it is fundamental to extracting meaningful details from data where high-quality inputs are unavailable or impractical to acquire.

### Applications of Super-Resolution
SR has far-reaching applications across various domains:
- **Medical Imaging:** Provides enhanced clarity in CT scans, MRI images, and X-rays, aiding more accurate diagnosis.
- **Satellite Imagery:** Improves spatial detail for monitoring environmental changes, urban growth, and disaster management.
- **Video Enhancement:** Plays a crucial role in restoring legacy video footage, streaming high-definition content, and ensuring a better viewing experience.
- **Security and Surveillance:** Enhances the resolution of footage from low-quality cameras, making it easier to identify objects and people.
- **Consumer Electronics:** Powers upscaling technologies in modern televisions, ensuring that content matches the resolution of high-definition displays.

### The Super-Resolution Problem
At its core, SR addresses the challenge of inferring missing high-frequency information from LR data. Mathematically, the relationship between the HR and LR image can be modeled as:
$$\ HR = f(LR) + \epsilon\$$
where $f$ represents the SR mapping function that is learned and $\epsilon$ is the reconstruction error. The goal of SR techniques is to learn an $f$ that minimizes $\epsilon$ while preserving fine details.

The main challenges in SR include:
1. **Ill-Posed Nature:** A single LR image can correspond to multiple possible HR images, making the problem inherently ambiguous.
2. **Trade-Off Between Quality and Computational Cost:** High-quality SR often requires complex models and heavy computational resources.
3. **Realism vs. Perception:** Balancing perceptually pleasing outputs with faithful HR reconstruction remains a significant hurdle.

With advances in machine learning and deep learning, SR techniques have evolved to overcome these limitations, delivering results that were previously unattainable.



---

## History of Classical Methods

Before the rise of deep learning, classical super-resolution methods relied on mathematical models and statistical assumptions. These methods laid the groundwork for modern SR techniques, offering insights into how high-frequency details could be recovered from low-resolution inputs. They can be broadly categorized as interpolation-based, reconstruction-based, and example-based approaches.

![](/assets/images/UCLAdeepvision/team17/bicubic.png)
*Figure 2: Different interpolation methods: black dot is interpolated estimate.*

### Interpolation-Based Methods
Interpolation methods, such as **nearest neighbor**, **bilinear**, and **bicubic interpolation**, estimate missing pixel values by averaging surrounding pixels. While these methods are computationally simple and fast, they often produce overly smooth results and lack the ability to recover finer textures or details.

### Reconstruction-Based Methods
Reconstruction-based techniques, like **iterative back-projection (IBP)**, utilize knowledge of the degradation process (e.g., downscaling and blurring) to iteratively refine the high-resolution output. These methods aim to maintain consistency between the LR input and the reconstructed HR image. Despite their promise, they are computationally expensive and sensitive to noise, limiting their applicability in real-world scenarios.

### Example-Based Methods
Example-based methods introduced the idea of leveraging external datasets to enhance SR. Techniques such as **sparse coding** involve training a dictionary of image patches to model the mapping between LR and HR pairs. These methods achieve better detail reconstruction compared to interpolation-based and reconstruction-based approaches but require extensive datasets and manual feature engineering.

### Transition to Deep Learning and Upsampling Techniques
Classical methods' limitations, such as reliance on handcrafted features, led to the adoption of deep learning. Early models like the **Super-Resolution Convolutional Neural Network (SRCNN)** demonstrated that neural networks could learn the mapping from LR to HR directly from data.

#### Key Components of Deep Learning SR:
1. **Feature Extraction:** Convolutional layers extract spatial features from the LR input.
2. **Non-Linear Mapping:** Hidden layers learn complex transformations to model high-frequency details.
3. **Upsampling Layers:** Deep learning-based SR methods typically include upsampling to increase the spatial dimensions of the image:
   - **Deconvolution (Transposed Convolution):** Applies learned filters to increase resolution while adding meaningful features.
   - **Sub-Pixel Convolution:** Rearranges features from depth to spatial dimensions, offering efficient upscaling.
   - **Interpolation + Convolution:** Combines simple interpolation with subsequent convolution layers to refine details.
4. **Reconstruction:** The final layers generate the HR output from learned features.

#### Example: SRCNN Workflow
1. The LR image is first interpolated to the desired resolution using bicubic interpolation.
2. Convolutional layers extract features from this upscaled image.
3. Additional layers map these features to high-resolution space.
4. The final layer reconstructs the HR image.

Deep learning SR models surpassed classical methods by automating feature learning and capturing complex patterns, paving the way for advanced architectures like GANs and attention mechanisms.

---
## Diffusion-Based Methods: Super-Resolution via Iterative Refinement (SR3)
SR3 (Super-Resolution via Iterative Refinement) is a novel approach to super-resolution using diffusion probabilistic models that was proposed by Chitwan Saharia et al. in 2021. It adapts these models for conditional image generation, achieving high-quality results through a stochastic denoising process.

SR3 operates in two key phases: Forward Diffusion and Reverse Denoising. It works as follows:
   1. **Forward Diffusion: Adding noise**
      
         - The forward process adds Gaussian noise to the HR target image step-by-step using a Markovian-based process. This transforms the HR image into pure noise over $T$ steps.

         - At the final step $T$, $y_T$ is approximately pure Gaussian noise.

   3. **Reverse Denoising: Removing noise**
      
        -  The reverse process begins with a noisy image $y_T \sim \mathcal{N}(0, I)$, and iteratively denoises it to produce $y_0$, the reconstructed HR image.

         - A U-Net-based denoising model $f_{\theta}$ is used to estimate the noise at each step. The denoising objective ensures the predicted noise is subtracted at every iteration:

            $$y_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( y_t - \frac{1 - \alpha_t}{\sqrt{1 - \gamma_t}} f_\theta(x, y_t, \gamma_t) \right) + \sqrt{1 - \alpha_t} z$$

            where $z \sim \mathcal{N}(0, I)$ is Gaussian noise added for stochasticity.

![](/assets/images/UCLAdeepvision/team17/forwardnoise_denoising.png)

For the model architecture, SR3 uses a modified U-Net backbone is used to process both the noisy HR image and the bicubic-upsampled LR image. These are concatenated channel-wise as input. Additionally, SR3 adds residual blocks and skip connections to improve gradient flow and learning efficiency. For efficient inference, SR3 sets the maximum inference budget to 100 diffusion steps, and hyper-parameter searches over the inference noise schedule.

As a result of these model architecture optimizations by SR3, the model is able to achieve state-of-the-art performance on multiple super-resolution tasks across datasets and domains (faces, natural images). 

![](/assets/images/UCLAdeepvision/team17/sr3onfacesuperresolutiontask.png)

As seen in figure 5 just above, the images that SR3 produced in this example for a 16x16 --> 128x128 face super-resolution task contain finer details, such as skin and hair texture texture, outperforming other GAN-based methods (FSRGAN, PULSE) and a regression baseline trained with MSE while also avoiding GAN artifacts like mode collapse. 

SR3 was able to achieve a fool rate close to 54%, meaning that it produces outputs that are nearly indistringuishable from real images. As a benchmark, the fool rate for the GAN-based method PULSE was 24.6% while the fool rate for FSRGAN was 8.9%, which showcases just how large of an improvement this is.

![](/assets/images/UCLAdeepvision/team17/sr3foolrates.png)

All in all, SR3 is a state-of-the-art diffusion-based super-resolution method that offers key advantages over GAN-based methods such as a stronger ability to generate sharp and detailed images, absence of GAN instability issues, such as mode collapse, due to probabilistic modeling, as well as efficiency due to its cascaded architecutre that allows for modular and parallel training.

## SwinIR: Image Restoration using Swin Transformer
SwinIR (Swin Transformer for Image Restoration) addresses two main limitations of traditional convolutional neural networks (CNNs) in image restoration: content-independent convolution kernels, which may not adapt well to varying image regions, and the limited ability to model long-range dependencies due to the localized nature of convolutions. Leveraging the Swin Transformer, which combines local attention mechanisms and a shifted window scheme for global dependency modeling, SwinIR integrates the best features of CNNs and Transformers to enhance both accuracy and efficiency in image restoration tasks.

SwinIR is structured into three key components:

Shallow Feature Extraction, where a single 3×3 convolutional layer maps the input image into a higher-dimensional feature space. This module focuses on preserving low-frequency image details, crucial for reconstructing stable and visually consistent images. This is followed by Deep Feature Extraction, which is built out of Residual Swin Transformer Blocks (RSTBs). Each RSTB contains multiple Swin Transformer Layers (STLs) for capturing local attention within non-overlapping image patches. The STLs use a shifted window partitioning mechanism to alternate between local and cross-window interactions, enabling long-range dependency modeling without introducing border artifacts. After each sequence of STLs, a convolutional layer is applied for feature enhancement, followed by a residual connection to aggregate features efficiently. The output of the RSTB integrates both spatially invariant convolutional features and spatially varying self-attention features, ensuring better restoration accuracy. Lastly, a High-Quality Image Reconstruction layer fuses shallow and deep features through a long skip connection to reconstruct the high-resolution image. Sub-pixel convolution layers to upscale the features [3].

![](/assets/images/UCLAdeepvision/team17/SwinIR_Architecture.png)

For superresolution tasks, they use the L1 pixel loss as the loss function, which is defined as:

$$
L = \lVert I_{RHQ} - I_{GT}\rVert_1
$$

where $$I_{RHQ}$$ is the high-resolution image reconstructed by SwinIR, and $$I_{GT}$$ is the ground truth high-resolution image [3]. 

SwinIR demonstrates state-of-the-art performance on classical Image Super-Resolution, outperforming CNN-based methods like RCAN and even other Transformer-based methods like IPT, achieving up to a 0.47 dB gain in PSNR on benchmark datasets, while maintaining a competitive runtime. SwinIR uses a comparatively small number of parameters (~11.8M) than other transformer based architectures like IPT, and even convolutional models [3]. 

![](/assets/images/UCLAdeepvision/team17/SwinIR_Performance.png)

## Super-Resolution Generative Adversarial Network (SR-GAN)
The inspiration behind the Super-Resolution Generative Adverserial Network, or SR-GAN, was to combine multiple elements of efficient sub-pixel nets that were conceived in the past with traditional GAN loss functions. To recap, the ordinary GAN architecture requires two neural networks - one called the generator and another called the discriminator. Here, the generator of the GAN serves to create new images from an input data sample while the discriminator also takes these same input data samples, but serves to distringuish between fake and real images produced by the GAN. The loss function to train this GAN is known as the min-max loss function, where the generator essentially tries to minimize this function while the discriminator tries to maximize it. The theory is that after training, the generator will produce images that the discriminator calculates as 50 percent change of being either fake or not. 

SR-GAN built upon this through developing a new loss function. Instead of just the min max loss function, SR GAN uses a perceptual loss function which consists of both an adverserial and content loss. The adversarial loss is the standard GAN loss, which pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. The content loss is motivated by perceptual similarity instead of similarity in pixel space. This is based on features extracted from a pre-trained deep network (such as VGG-19) to measure the perceptual similarity between the generated and ground truth images. This loss captures high-level features like textures and structures, which are more aligned with human visual perception. This deep residual network is then able to recover photo-realistic textures from heavily downsampled images on public benchmarks. [1]


Formally they write the perceptual loss function as a weighted sum of a (VGG) content loss and an adversarial loss component as:

$$
\[
L_{\text{SRGAN}}(Y, \hat{Y}) = L_{\text{content}}(Y, \hat{Y}) + \lambda L_{\text{adv}}(G, D)
\]
$$

where the total loss for the SR-GAN is a weighted sum of the content loss and adversarial loss and lambda is a hyperparameter. 

Below is an image and description of the specific architecture of the generator and discriminator. 

![](/assets/images/UCLAdeepvision/team17/SR-GAN-Arch.JPG)

Firstly, we will go over the generator, whose main objective is to upsample Low Resolution images to super-resolution images. The main difference between an ordinary GAN and SR-GAN is that the generator takes in a Low Resolution input and then passes an image through a Conv layer. This is different from an ordinary GAN because ordinary GANs takes in a noise vector. The generator then employs residual blocks, where the idea is to keep information from previous layers and allow the network to choose features more adaptively. The discriminator is follows the same standard architecture of GANs.

Below is a result of how the SR-GAN performs relative to other Super Resolution models.
![](/assets/images/UCLAdeepvision/team17/SR-GAN-Results.JPG)

As we can see from the images and table published in "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" [1], SR-GANs seem to outperform it counterparts. From the table PSNR is a metric used to measure the quality of an image or video by comparing a compressed or noisy image with the original one. It is based on the ratio of the peak signal (maximum possible pixel value) to the noise (difference between the original and the processed image). A higher value the better, and as we can see SR-GAN and SR-Resnet outperforms all other models.

---
[1] Ledwif, Christian, et al. "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" 2017

[2] Saharia, Chitwan, et al. "Image Super-Resolution via Iterative Refinement." arXiv preprint arXiv:2104.07636, 2021.

[3] Liang, Jingyun, et al. "SwinIR: Image Restoration Using Swin Transformer." *arXiv preprint arXiv:2108.10257* (2021).

---
