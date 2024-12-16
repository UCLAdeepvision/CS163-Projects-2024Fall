---
layout: post
comments: true
title: Image-to-Image Style Transfer
author: Leon Liu, Rit Agarwal, Tony Chen, Tejas Kamtam
date: 2024-12-13
---

> Recent advances in deep generative models have enabled unprecedented control over image synthesis and transformation, from translating between visual domains to precisely controlling spatial structure or appearance in generated images. This report traces a few seminal developments in architectures and training methodologies that have made these capabilities possible, from GAN-based approaches like CycleGAN to modern diffusion-based techniques that leverage pre-trained Stable Diffusion models to enable fine-grained control through image conditioning.

<!--more-->
{: class="table-of-content"}
* [Introduction](#introduction)
* [GANS](#gans)
  * [CycleGAN](#cyclegan)
  * [StyleGAN](#stylegan)
* [Diffusion-Based Methods](#diffusion-based-methods)
  * [Diffusion](#diffusion)
  * [Stable Diffusion](#stable-diffusion)
  * [ControlNet](#controlnet)
  * [Ctrl-X](#ctrl-x)
* [References](#references)
* [Implementation](#implementation)

{:toc}

# Introduction

Image-to-image translation and controllable generation techniques have emerged as crucial tools in computer vision and graphics, addressing the fundamental challenge of transforming images while preserving specific attributes. These methods serve various practical applications, from artistic style transfer to solving critical problems in robotics and computer vision.

One significant application is bridging the "sim to real" gap between synthetic and real-world data. Modern robotics and computer vision systems may take advantage rely on training data generated from simulators like Unity or Unreal Engine because they provide perfect ground truth labels and more practical data generation capabilities rather than deploying in the real world. Sandboxing a model in a virtual environment is also a lot safer especially during early phases of training. However, models trained purely on synthetic data often struggle when deployed in real-world scenarios due to differences in lighting, textures, and object appearances. Image-to-image translation techniques can help bridge this domain gap by translating synthetic images to appear more realistic while preserving critical structural information like object positions, shapes, and semantic labels (e.g. SimGen).

Beyond simulation-to-real translation, these techniques enable broader forms of controllable image generation. For instance, in architectural visualization, rough sketches can be transformed into photorealistic renderings, or in fashion design, clothing items can be visualized on different body types while maintaining structural integrity (ex. Outfit Anything). Medical imaging applications benefit from the ability to transfer appearances between imaging modalities (e.g., MRI to CT) while preserving anatomical structure.

Beyond practical usage, image generation is probably the most popular usage of models like Stable Diffusion, and image conditioning is desirable for users seeking finer grain control on spatial information or appearance.

The field has evolved from early approaches using Generative Adversarial Networks (GANs) like CycleGAN, which learned direct mappings between image domains, to more recent diffusion-based methods that offer greater control, stability, and generalization. Modern techniques like ControlNet and Ctrl-X build upon large pretrained diffusion models to enable precise spatial control over generation while maintaining high image quality. These advances have made image-to-image translation more practical and widely applicable.

---

# GANs
Generative Adversarial Networks (GANs) introduced a novel approach to generative modeling through an adversarial game between two networks: a generator that creates fake samples and a discriminator that tries to distinguish real from fake samples. The generator learns to produce increasingly realistic images by trying to fool the discriminator, while the discriminator simultaneously improves its ability to detect generated images. This adversarial training dynamic helps GANs capture complex data distributions without explicitly modeling them.
Early GANs showed impressive results for unconditional image generation but struggled with mode collapse (generating a limited variety of samples) and training instability. Various architectural and training improvements were proposed, leading to more stable variants like DCGAN and later StyleGAN. However, the challenge of controlled generation remained - how could we transform images between domains while preserving specific attributes?

## StyleGAN
StyleGAN more or less began the march towards high quality style transfer through a novel style-based generator architecture, enabling control over style micro and macro feature modulation. Inspired by style transfer techniques, StyleGAN separates high-level attributes (e.g., pose, identity) from fine stochastic details (e.g., freckles, hair textures) in 3 main levels: coarse, middle, and fine. Along with new noise modulation techniques, this innovative design not only enhances the quality of generated images but also introduces intuitive, scale-specific control of image generation.

The core idea of StyleGAN is to treat the image generation process like style transfer, where "styles" govern the high-level attributes of an image. Traditional GAN generators embed latent vectors directly into the input layer of the network, acting as a black box that entangles features such as pose, lighting, and textures. StyleGAN decouples these factors by embedding the latent code into an intermediate latent space $$W$$, followed by the injection of styles at multiple layers of the synthesis network

This architecture provides:
- **Unsupervised** disentanglement of high-level attributes from stochastic features.
- **Scale-specific control**, allowing edits to pose, structure, and fine details independently.
- **Improved image quality**, with smoother latent space interpolation and better latent disentanglement.

The paper presents these novel techniques through a GAN trained on the <a href="https://github.com/NVlabs/ffhq-dataset">FFHQ dataset</a> (Flickr-Faces-HQ dataset) which contains 52,000 high-quality $$512\times512$$ images of human faces.


### Architecture

### Traditional vs. Style-Based Generators

In traditional GANs, the latent code $$z$$ directly influences the entire synthesis process, leading to entangled features. StyleGAN departs from this by introducing an intermediate latent space $$W$$ and a style-based generator.

Key Components:
1. Mapping Network
    - An MLP with 8 fully connected layers. Outputs $$\omega$$, a disentangled representation of $$z$$

2. Synthesis Network
    - Begins with a constant learned tensor (4×4×5124×4×512) instead of a latent vector.
    - Applies AdaIN to adjust feature maps at each layer.

3. Noise Injection
    - Injects uncorrelated noise after each convolution, enabling stochastic details.

![StyleGAN Architecture Comparison]({{ '/assets/images/22/StyleGAN.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*StyleGAN Architecture Comparison [<a href="#references">9</a>]*


Instead of feeding the latent vector $$z$$ directly into the generator, StyleGAN introduces:

**Mapping Network**: An 8-layer MLP that transforms $$x \in Z$$ into an intermediate latent space $$w \in W$$. This disentangles the latent factors and removes dependency on the probability density of $$z$$

**Adaptive Instance Normalization (AdaIN)**: Each convolution layer uses styles  to scale and shift feature maps, offering explicit control over attributes like pose, color, and texture. The AdaIN operation is defined as

$$
\text{AdaIn}(x_i, y)=y_{s,i}\frac{x_i-\mu}{\sigma(x_i)}+y_{b,i}
$$

![StyleGAN Noise Modulation]({{ '/assets/images/22/StyleGAN_noise.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*StyleGAN Noise Modulation [<a href="#references">9</a>]*

In the image above, the authors show how the noise modulation via the AdaIN operation at every layer of the generator can be used to control the noise at each layer. The noise controls very fine grained details like freckles, hair placement, and background textures without affecting the high level features like pose, identity, and facial structure.

**Scale-Specific Styles**: Different layers control different aspects of the image:
Coarse layers influence pose and shape.
Middle layers affect features like facial structure.
Fine layers refine textures and details like hair or skin tone.


![StyleGAN Style Modulation]({{ '/assets/images/22/StyleGAN_styles.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*StyleGAN Style Modulation Results [<a href="#references">9</a>]*

For example, in the image above, the authors pass an input/base image from Source B and apply varying levels of "style" from images of Source A. The resulting images seem to inherit the style properties of their Source A counterpart overlayed on the base image from Source A. This introduces both structural adherence to the base image and controllable style properties of a style image. That too at multiple levels of feature prominence.

### Limitations

Although StyleGAN has proven to be a powerful tool for style transfer, it does have some limitations. Specifically, interpretability is a continued concern. Although style and noise modulation are available at multiple levels and scale with features, the process is still random as the style and noise modulation is done by tuning Gaussian noise. Without a specific text-based guidance system (or other specifiable method), it is mostly up to trial and error to achieve specific style properties during style transfer. Additionally, the model tends to be heavily biased toward the dataset demographic representation. There stands to be improvement in high quality, diverse human face datasets

---

## CycleGAN

CycleGAN addresses the unpaired image-to-image translation problem: learning to translate between domains (e.g., horses to zebras, summer to winter scenes) without paired training examples. 

![CycleGAN Examples]({{ '/assets/images/22/CycleGANExamples.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Example translations from domain to domain creates by CycleGAN [<a href="#references">6</a>]* 

The key insight is that while we may not have direct pairs showing how each image should be translated, we can enforce cycle consistency translating an image to the target domain and back should recover the original image, and thus we have training data where the output (after passing it back through both models) target is the input.

### Architecture


![CycleGAN]({{ '/assets/images/22/CycleGAN.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*CycleGAN Architecture [<a href="#references">6</a>]*

The architecture consists of two generator-discriminator pairs:

Generator G: Translates from domain X to domain Y

Generator F: Translates from domain Y back to domain X

Discriminator $$D_y$$: Distinguishes real Y domain images from generated ones

Discriminator $$D_x$$: Distinguishes real X domain images from generated ones

The cycle consistency is enforced through two pathways:

Forward cycle: x-> G(x) -> F(G(x)) ≈ x

Backward cycle: y -> F(y) -> G(F(y)) ≈ y

$$
L_{cyc}(G, F) = E_{x~p_{data}(x)} [\|F(G(x)) - x\||_1] + E_{y~p_{data}(y)} [\|G(F(y)) - y\|_1]
$$

The loss function combines:

Adversarial losses that encourage the generated images to match the target domain's distribution and cycle consistency losses that ensure the original image can be reconstructed after translation in both directions:

$$
L(G, F, D_X, D_Y) = L_{GAN}(G, D_Y, X, Y) + L_{GAN}(F, D_X, Y, X) + \lambda L_{cyc}(G, F)
$$

Here, $$L_{GAN}$$ is the same minimax adversarial loss where each generator must fool the discriminator jointly optimized on to recognize the generated image of an output domain and real images from that domain. There are two, because there are two generators for each domain. Finally the cycle consistency loss 

### Limitations
Because of this paired cycle relationship, CycleGAN learns a specific pair of domain to domain translations that it is trained on and thus is not a flexible method to handle translations between a diverse set of domains. Additionally, GAN architectures have become less popular as diffusion-based methods like Stable Diffusion have arisen. 

---

# Diffusion-Based Methods

## Diffusion 
Generative modeling fundamentally aims to learn and sample from complex data distributions $$p(\mathbf{x})$$, such as the distribution of natural images. Earlier approaches like Variational Autoencoders (VAEs) tackle this by learning a mapping between the data and a simpler latent space. A VAE builds upon previous encoder-decoder methods to intentionally encoding data points into a latent distribution $$q(z|\mathbf{x})$$ that is similar to a target simple prior $$p(z)$$ (such as a Gaussian distribution) and decoding samples from this distribution back to data space $$p(\mathbf{x}|z)$$ with the goal of maximizing the likelihoods of the target data distribution. A KL term is used to maximize the similarity of the latent distribution conditioned on the input image distribution to the simple prior in order to be able sample latent vectors (not encoded from any particular ground truth image) directly from the prior which could be a Gaussian diffusion to feed through the decoder and generate a novel image. This approach struggles with complex distributions as it must learn to map between very different domains in a single step.

Diffusion models take a different approach to approximating the target image distribution but fundamentally still have the goal of generating an image from pure noise. Rather than trying to directly model the complex data distribution, they break down the generation process into a sequence of small denoising steps that gradually transform pure noise into samples from the target distribution. This is motivated by the observation that while $$p(\mathbf{x})$$ may be complex, the transition between slightly different noise levels can be well-approximated by simple Gaussian distributions.

This means that the model is only responsible for predicting the Gaussian noise added from each step rather than going directly predicting a target image and thus does not have to approximate our target image distribution. Our goal is fundamentally the same which is to go from a noised image to a slightly less noised image until we eventually reach an image. This is the intuition and we will go further into details in the Backward Process section.

### Forward Process: 
*How can we create training data and noise an image?*

The forward process defines adding a sequence of progressively noisier distributions (based on a variance/noise scheduler which determines the amount of noise to add at a timestep), starting from the data distribution and ending at Gaussian noise. Given a data point $$\mathbf{x}_0$$, we add noise over $$T$$ timesteps according to:

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

where $$\beta_t$$ is a variance schedule that controls the noise level at each step. After adding noise over many many timesteps, the final $$x_T$$ is a Gaussian distribution.

This process is a Markov process in that each step depends only on the previous state. This allows us to arbitrarily sample any timestep directly:

$$
q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_0, (1-\alpha_t)\mathbf{I})
$$

$$
\alpha_t = \prod_{s=1}^t (1-\beta_s)
$$

This is achieved through the reparameterization:

$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_0 + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

This is to say that at each step we sample noise according to our noise scheduler to gradually noise an image, however thanks to the reparameterization trick, we can arbitrarily sample the noised image at a particular timestep using our precomputed cumulative products. 

#### Code Example
This is taken from Professor Zhou's CS163 Assignment 4:
```python
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, 0)
sqrt_alphas_cumprod = alphas_cumprod.sqrt()
sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

@torch.no_grad()
def forward_diffusion_sample(x_0, t):
    """
    Perform the forward diffusion process
    x0: input image of shape [B, C, H, W]
    t: input timesteps of shape [B]
    use the provided "get_index_from_list" function to get values at timestep t
    returns:
    x_t: noisy image at time step t of shape [B, C, H, W]
    noise: the generated gaussian noise
    """
    x_t = None
    noise = None
    
    sqrt_alpha_cumprod = get_index_from_list(sqrt_alphas_cumprod, t)
    sqrt_one_minus_alpha_cumprod = get_index_from_list(sqrt_one_minus_alphas_cumprod, t)
    
    noise = torch.randn_like(x_0) 
    x_t = x_0 * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
    
    return x_t, noise
```

![ForwardPassExample]({{ '/assets/images/22/MNIST_Forward_Process.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

This forward pass example comes from Assignment 4, in which a sample image is noised using the noise schedule from above.

### Backward Process 
*How do we undo the noise?*

The backwards process allows us to denoise data and recover an image from a noised image by approximating $$p(x_{t-1}|x_t)$$. 
Using Bayes' rule, we can derive that the backward process is also Gaussian:

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q(\mathbf{x}_t,\mathbf{x}_0), \sigma^2_t\mathbf{I})
$$

To calculate $$\boldsymbol{\mu}_q$$, we need $$\mathbf{x}_0$$. From the forward process, we know:

$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_0 + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}
$$

Rearranging this to solve for $$\mathbf{x}_0$$:

$$
\mathbf{x}_0 = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \sqrt{1-\alpha_t}\boldsymbol{\epsilon})
$$

This shows that if we know the noise $$\boldsymbol{\epsilon}$$ that was added to get to $$\mathbf{x}_t$$, we can recover $$\mathbf{x}_0$$. This motivates the approach of training a neural network $$\epsilon_\theta$$ to predict the noise rather than directly predicting $$\mathbf{x}_0$$ or $$\mathbf{x}_{t-1}$$,

$$
\epsilon_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}
$$

Once we have this prediction, we can estimate $$\mathbf{x}_0$$ as:

$$
\hat{\mathbf{x}}_0 = \frac{1}{\sqrt{\alpha_t}}(\mathbf{x}_t - \sqrt{1-\alpha_t}\epsilon_\theta(\mathbf{x}_t, t))
$$

And then plug this back into our original expression for $$\boldsymbol{\mu}_q$$:

$$
\boldsymbol{\mu}_q(\mathbf{x}_t,\hat{\mathbf{x}}_0) = \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\hat{\mathbf{x}}_0 + \frac{\sqrt{1-\beta_t}(1-\alpha_{t-1})}{1-\alpha_t}\mathbf{x}_t
$$

This formulation has several advantages:

Predicting noise is easier than predicting images directly
The scale of the noise prediction target remains consistent throughout training
We can generate unlimited $$(\mathbf{x}_t, \boldsymbol{\epsilon})$$ pairs for training using our forward process

The complete backward process then becomes:

$$
p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma^2_t\mathbf{I})
$$

where $$\boldsymbol{\mu}_\theta$$ is computed using our noise prediction network $$\epsilon_{\theta}$$.

### Sampling 

There are multilple approaches to sampling. But one such approach is DDPM:
#### Code Example
This is taken from Professor Zhou's CS163 Assignment 4:
```python
@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    x: input image at timestamp t [B, C, H, W]
    t: the timestamp [B]
    """
    x_denoised = None

    alpha_t = get_index_from_list(alphas, t)
    alpha_bar_t = get_index_from_list(alphas_cumprod, t)
    
    # Calculate predicted noise
    pred_noise = model(x, t)
    
    # Calculate coefficient for predicted noise
    noise_coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    
    z = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
    
    beta_t = get_index_from_list(betas, t)
    
    sigma_t = torch.sqrt(beta_t)
    
    x_denoised = (1 / torch.sqrt(alpha_t)) * (x - noise_coeff * pred_noise) + sigma_t * z

    return x_denoised
```

The approximated original image is derived from the equation above but then the image is renoised with $$\sigma_t$$ to account to recover get back a partially denoised image. The sampling can then be performed iteratively rather than in one shot. Intuitively, it's easier to predict small changes to move closer to a less noise image. 

![Sample Backward Process on MNIST]({{ '/assets/images/22/MNIST_Backward_Process.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

### Training and Simplified U-Net Model

The backbone model used to predict the noise is a U-Net which resembles an encoder-decoder model with additional skip connections (kind of similar to residual connections) concatenating outputs of encoder blocks to decoder blocks of matching dimensions such that the model diagram resembles a "U" [<a href="#references">2</a>]. 

The model is also conditioned on the current timestep which helps inform how "noisy" the input image will be.
An addition made to the standard U-Net is a time embedding which converts the low frequency timestep information to high frequency sinusoidal embeddings. These embeddings are projected into the output channel dimension for each block using a linear layer and added to each spatial location / pixel.

During training, random timesteps are sampled and the ground truth images are noised using the forward process to noise the image and the model tries to predict the noise added to the image, to in one-shot recover the ground truth. L1 or MSE loss are some example loss functions which can be used to optimize the model.

![Diagram of a U-Net]({{ '/assets/images/22/Unet.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*U-Net Model Diagram [<a href="#references">2</a>]*

There are a few more additions made to the U-Net used in Stable Diffusion such as self-attention and cross-attention which we'll go in the next section, but this is the general idea.

---

## Stable Diffusion

While the basic diffusion framework is powerful, operating in pixel space presents significant computational challenges. Stable Diffusion [<a href="#references">3</a>] introduces several key innovations that make diffusion models more practical while maintaining generation quality.


![Stable Diffusion (LDM)]({{ '/assets/images/22/StableDiffusion.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Diagram of Stable Diffusion / Latent Diffusion's overall process [<a href="#references">3</a>]*

### Latent Space Diffusion

The core insight of Stable Diffusion is that we don't need to perform diffusion in high-dimensional pixel space. 

We can instead train an VAE to encode our target images into latent space and recover the original image with decoding. 

Then because the lower-dimensional latent space has been trained to represent compact, rich features of the image, we can perform diffusion on latent space instead of pixel space.

### AutoEncoder (VAE)

The compression is achieved using a Variational Autoencoder (VAE) that consists of:

An encoder $$\mathcal{E}$$ that maps images $$\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$$ to latents $$\mathbf{z} \in \mathbb{R}^{h \times w \times c}$$
A decoder $$\mathcal{D}$$ that reconstructs images from these latents

In general, the encoder and decoder objective is to maximize a variational lower bound on the data likelihood, which helps train the model to maximize the likelihood of generating the target image data. The Stable Diffusion paper proposes a combination of perceptual loss and patch-based adversarial loss to enforce local realism. To avoid high-variance latent spaces, the method also uses a KL penalty to push the learned latent towards a standard Gaussian distribution.

On a high level, the goal of the VAE is to learn a good efficient representational space to then train a diffusion model on the latent representations of the target images.

### U-Net

The neural backbone of the Stable Diffusion model is a U-Net [<a href="#references">2</a>], which has also been used in Diffusion in pixel space. Similar to diffusion, the objective is to accurately predict the noise added, except the noise is added to the latent instead of the image directly. 

The U-Net model is conditioned with the timestep as before.

The previously discussed U-Net architecture was simplified compared to the architecture used within Stable Diffusion.
Specifically, the U-Net modifies the Block Components to use a ResNet block
and also adds additional cross-attention steps to attend to textual embeddings from a provided prompt. Self-attention is also performed at selected blocks to enhance self-consistency by allowing information to mix across the spatial dimension.
More on this, in the Ctrl+X section. 

Aside: A brief summary of attention:
In self-attention, a Query, Key, and Value vector is produced from every spatial features in your image.
The objective is for each feature to extract relevant context from all other features and add it onto the feature's context. In order to to do this, each Query vector represents what a feature is looking for, and each Key vector represents what a feature has. In practice these vectors are actually matrices of number of queries x query dim and number of keys x query dim. Then a dot product is taken to compute the similarities $$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)$$ and rescaling and softmax (along the key dimension) is applied to get a probability distribution for each query which is multiplied with the values to take a weighted average of the values weighted on how "relevant" they are. The values represent "what context a feature provides" and so relevant values should be extracted from relevant features and added into a feature's context. 

In self-attention, the queries, keys, and values are all produced from the spatial features but in cross-attention, the keys and values originate from other embeddings (such as textual embeddings or even image embeddings). This allows the spatial image features to attend to textual embedding information and add appropriate context to each feature.

The order of layers in the forward pass for a block is now: 1. ResNet block, 2. Self-Attention, 3. Cross-Attention. These all use residual connections to add information on to the residual stream. Also, self-attention and cross-attention are not necessarily performed on every block.

Details can be found in the Stable Diffusion repo's `stable-diffusion/ldm/modules/diffusionmodules/model.py` file here
[here](https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py).

### Text-Conditioning

A common model used to produce textual embeddings for textual prompting is CLIP.
CLIP's text encoder is based on a transformer architecture that converts a textual into a semantic embedding space which is then used in cross-attention to allow spatial features to extract relevant semantic information from the textual prompt during the diffusion process.



![Stable Diffusion Results]({{ '/assets/images/22/SDResults.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Sample outputs from model conditioned on user prompts [<a href="#references">3</a>]* 

### Training Details

Since Stable Diffusion is composed of a VAE model and the Latent Diffusion Model which relies on the VAE's latent represntation, the VAE must be trained first.

**KL-regularized Autoencoder Training**

The first stage trains the autoencoder (VAE) with a combination of reconstruction loss, KL-divergence regularization, adversarial loss, and perceptual loss.

The KL term which they refer to as KL regularization $$\mathcal{L}_{\text{reg}}$$ encourages the learned latent distribution to be close to a standard normal distribution, which is beneficial for the diffusion process. The perceptual loss uses deep features from a pretrained VGG network to maintain semantic consistency, while the adversarial loss $$\mathcal{L}_{\text{adv}}$$ helps preserve realistic high-frequency details. 

The idea behind perceptual loss is that the feature space in a pretrained VGG model better encode higher level concepts and semantically significant features and thus, it is important to maintain similar semantic consistency between the original image and the VAE encoding-decoding of it.

Adversarial loss is another patch-based descriminator that is optimized to differentiate original images from reconstructions. This is similar to the discriminator used in GANs however this discriminator is intended to specifically enforce local realism.

One way to view this is that these metrics are more meaningful and correlated to human perceptions compared to pixel-space losses like MSE and L1 loss which do not consider how the pixels relate to each other and rather treats each pixel as an independent output.

$$\mathcal{L}_{\text{Autoencoder}} = \min_{E,D} \max_{\psi} \mathcal{L}_{\text{rec}}(\mathbf{x}, D(E(\mathbf{x}))) - \mathcal{L}_{\text{adv}}(D(E(\mathbf{x}))) + \log D_{\psi}(\mathbf{x}) + \mathcal{L}_{\text{reg}}(\mathbf{x}; E, D)$$

where:
- $$E$$ and $$D$$ are the encoder and decoder networks
- $$\mathcal{L}_{\text{rec}}$$ is a reconstruction loss 
- $$\mathcal{L}_{\text{adv}}$$ encourages realistic reconstructions through adversarial training
- $$D_{\psi}$$ is a discriminator that helps maintain perceptual quality (the perceptual loss)
- $$\mathcal{L}_{\text{reg}}$$ is a KL regularization term and is usally weighted to be very small

**Latent Diffusion Training**

Once the VAE is trained, the diffusion model is trained in the learned latent space. The objective follows the standard diffusion training framework but operates on encoded latents:

$$\mathcal{L}_{\text{LDM}} := \mathbb{E}_{x,c,\epsilon\sim\mathcal{N}(0,1),t}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(c))\|_2^2\right]$$

where $$z_t$$ is the noised latent at timestep $$t$$ obtained by encoding the input image $$x$$ into the latent space and adding noise according to the noise schedule (same as the previously described forward process):

$$z_t = \sqrt{\alpha_t}z_0 + \sqrt{1-\alpha_t}\epsilon$$

Here, $$z_0$$ is the clean encoded latent from the VAE, $$\epsilon$$ is random Gaussian noise sampled as the ground truth noise that was added, and $$\alpha_t$$ follows the noise schedule that determines how much noise is added at each timestep $$t$$. The model $$\epsilon_\theta$$ learns to predict this added noise given the noised latent $$z_t$$, the timestep $$t$$, and text conditioning $$\tau_\theta(c)$$ from the CLIP text encoder where $$\tau_\theta$$ is the encoder itself and the text prompt is $$c$$.

The model is trained to minimize the L2 distance between the predicted and actual noise. This simple objective, combined with the carefully designed forward noising process, allows the model to learn how to gradually denoise images while being conditioned on text descriptions.

### Limitations

One limitation of the Stable Diffusion base model is the inability to provide good spatial control with prompting alone. If a user wants to input an image as conditioning to add spatial control, the SD model with only cross-attention to textual prompting doesn't allow for sufficent control.

Due to being open source, many methods have been built around Stable Diffusion and there are finetunes that aim to improve certain aspects of image quality or produce images in a particular style.

The limitations around spatial control, motivated the development of methods like ControlNet and Ctrl-X, which we will discuss next. These approaches build upon Stable Diffusion's foundation while introducing new mechanisms for precise spatial conditioning and appearance transfer. 

Additionally, it's through methods that add spatial conditioning like ControlNet and Ctrl-X that it is possible to extract structural information from an input image to create a new stylized image that retains the original image's structure, thus performing style transfer.

---

## ControlNet

ControlNet introduces a principled approach for adding spatial conditioning to large pretrained text-to-image diffusion models. The key insight is to preserve the capabilities of the original model while learning how to incorporate structural conditions through a trainable copy of the model backbone. This enables pixel-perfect control over generation while maintaining the high quality and semantic understanding of the base model.

The intuition behind ControlNet is that we want to "control" the generation process by injecting structural information (like edges, depth maps, or pose) while preserving the rich semantic and appearance knowledge learned by the base model. This is challenging because directly modifying or fine-tuning the base model could degrade its performance. Instead, ControlNet creates a parallel network that learns to process and then integrate condition information into a frozen base Stable Diffusion model.

![ControlNetExample]({{ '/assets/images/22/ControlNetExample.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*ControlNet Samples* [<a href="#references">4</a>]

### Adapter
![ControlNet]({{ '/assets/images/22/ControlNetFull.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*ControlNet on the encoder blocks and middle block of Stable Diffusion U-Net* [<a href="#references">4</a>].

The pretrained U-Net backbone in the Stable Diffusion model is kept the same and the weights are frozen during the training of the adapter model. This prevents the loss / or retraining of the model's original image generation capabilities since the primary goal is to add in additional structural information without harming the quality of the model.

Thus, a copy is made from the Stable Diffusion U-Net backbone's encoding layers to form the ControlNet model.

Given some 512x512 pixel-space image containing spatial conditioning information, a small CNN consisting of four convolutional layers with 4x4 kernel and 2x2 strides are used to produce a 64 by 64 feature space vector to match that of the latent image produced by Stable Diffusion's VAE. Now that the shapes match,t he noised latent at a timestep is added onto this conditioning information and inputted into the first encoder layer of the ControlNet.

Since the ControlNet Encoder architecture exactly matches that of the SD Encoder layers, the outputs are likewise connected with skip connections to the corresponding Decoder layers of the original model. However unlike the original encoding layers, the ControlNet Encoder outputs are passed through zero initialized 1x1 convolutions first. This serves the purpose of initially zeroing out all the outputs of the encoder model to avoid ruining the image quality of the SD model when the ControlNet hasn't been trained.

As the ControlNet model trains, the zero convolutions will gradually become non-zero and conditioning information will be fed through once the ControlNet encoder layers learn the necessary weights to add on meaningful structural conditioning features to the decoder layers.

### Training 

ControlNet is trained on paired data consisting of images and their corresponding condition inputs (like edge maps, depth maps, or segmentation masks). The training process follows a similar approach to Stable Diffusion but focuses only on learning the ControlNet adapter while keeping the base model frozen.

For each training step, an image is first encoded into the latent space using the pretrained VAE encoder from Stable Diffusion. The condition image (such as a canny edge map) is processed through a small encoder network consisting of four convolution layers to match the spatial dimensions of the latent space. This creates the paired training data of (latent, condition) that ControlNet learns from.

The forward diffusion process is then applied to add noise to the latent according to the noise schedule. The ControlNet model along with the frozen Stable Diffusion U-Net backbone learns to predict this noise conditioned on both the noisy latent and the processed condition image. This mirrors the original Stable Diffusion training, except now the model has access to explicit structural information through the condition pathway.

Notably, the loss function remains the simple L2 loss between the predicted and ground truth noise:

$$\mathcal{L} = \mathbb{E}_{z_0,t,c_t,c_f,\epsilon\sim\mathcal{N}(0,1)} \left[||\epsilon - \epsilon\theta(z_t, t, c_t, c_f)||_2^2\right]$$

This straightforward objective works because the zero convolutions naturally modulate how much conditioning information to add, and there's no need for additional loss terms to balance structure preservation versus image quality since the architecture handles this automatically through the gradual training of the zero convolutions.

The model is trained using a relatively small dataset compared to the original Stable Diffusion model (around 50k pairs versus billions of images). This is possible because the task is more constrained as instead of learning to generate images from scratch, ControlNet only needs to learn how to integrate structural conditions into an already capable generative model. 

A key aspect of training is that 50% of the time, the text conditioning from the base model is replaced with an empty string. This encourages ControlNet to learn to extract semantic meaning directly from the condition images rather than relying solely on the text prompts. As a result, the trained model can effectively utilize structure conditions even without text prompts, while still being able to combine structural and textual conditioning when both are provided.

This training approach, combined with the zero convolution architecture, allows ControlNet to learn strong spatial control while fully preserving the generation capabilities and image quality of the base model. The result is a highly flexible conditioning mechanism that can be trained efficiently on different types of structural conditions.

### Limitations

While ControlNet provides effective spatial control over image generation, it faces two significant limitations: computational overhead and training data requirements.

#### Computational Overhead

A major limitation of ControlNet is its computational inefficiency during inference. The method requires running a complete copy of the Stable Diffusion encoder at every sampling timestep, effectively doubling the computation needed for the encoding pathway. This is because the ControlNet encoder must process both the conditioning information and the noised latent at each denoising step.

Unlike simpler conditioning approaches that might preprocess structural information once, ControlNet needs to repeatedly process the evolving noised state alongside the condition. This means that for a typical sampling process with 50 timesteps, the additional encoder copy must run 50 times. The computational burden becomes even more pronounced when using multiple ControlNet modules for different types of conditions, as each one requires its own complete encoder copy.

From this, methods like T2I-Adapter apply ControlNet's fundamental ideas of adding an adapter model but aimed for more computational efficiency by simplifying the encoder architecture and making it time independent. However even then, there are still limitations of having to train at all, which motivates a the paradigm of training-free guidance.

#### Training Data Requirements

A second fundamental limitation lies in ControlNet's reliance on paired training data. The model requires corresponding pairs of ground truth images and their structural representations (like edge maps, pose keypoints, or segmentation masks). While this data can be readily obtained for certain conditions using pretrained extractors (such as Canny edge detection or pose estimation models), it becomes challenging for more complex or artistic structural formats.

For instance, collecting paired data between:
- Hand-drawn sketches and photorealistic images
- 3D models and their 2D renders from specific viewpoints
- Artistic or stylized structural representations and natural images

becomes significantly more difficult as these pairings cannot be easily automated or extracted from existing images. This limitation restricts ControlNet's applicability to primarily those structural conditions where automated extraction tools exist or where large paired datasets can be feasibly collected.

These limitations motivate the development of more efficient, training-free approaches that can work with arbitrary structural conditions without requiring paired training data. Methods like Ctrl-X specifically address these challenges by eliminating the need for additional training while reducing the computational overhead during inference.

---

## Ctrl-X

Compared to approaches that use adapters, Ctrl-X takes a significant shift in approach to controllable image generation by eliminating the need for additional training or optimization steps. It is also guidance-free (an approach not covered in this report), but essentially doesn't require inference time backpropagation (defining losses to steer generation, e.g.g moving the centroid of an object). Unlike ControlNet which requires paired training data and adds computational overhead, Ctrl-X achieves structure and appearance control by leveraging the inherent capabilities of pretrained diffusion models through attention between intermediate features produced during the denoising process of a structure image, appearance image, and output image.

![Ctrl-X Samples]({{ '/assets/images/22/CtrlXExamples.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Sample generations from Ctrl-X* [<a href="#references">5</a>]

### Training-Free

The key insight that makes Ctrl-X possible is that the rich feature representations already learned by diffusion models can be directly manipulated to achieve controlled generation. 

This approach offers several advantages over previous methods:

Firstly, no additional training or fine-tuning is required, making it immediately applicable to any pretrained diffusion model. There still is comuptational overhead in having to pass the structural image and appearance image through the Stable Diffusion model as well.

Thanks to not needing training, there is no need for training data, and the generalizability of this approach means that arbitrary structural conditions can be used without having trained some adapter on this specific form of pairwise data.

### Methodology

The core idea behind Ctrl-X is to use the semantic correspondence capabilities of self-attention layers to transfer appearance while maintaining structural integrity. Given a structure image and an appearance image, the goal is to generate a new image that preserves the structural information of the first while adopting the style and appearance of the second.

The process works by carefully manipulating three types of features within the diffusion model:

1. Structural features from convolution layers
2. Self-attention patterns that encode semantic relationships
3. Appearance statistics that capture style information

#### Feature Manipulation Pipeline

For any given denoising timestep t, the process follows several key steps:

1. Both the structure and appearance images are noised using the forward diffusion process according to the current timestep
2. These noised images are passed through the base diffusion model to extract features
3. Structural features are injected through convolution layers replacing the features of the outputs layer
4. Appearance transfer is performed through attention based normalization
5. Self-attention patterns are injected from the structural features used to reinforce structural alignment to the structural image

![CtrlXPipeline]({{ '/assets/images/22/CtrlXPipeline.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Ctrl-X Pipeline* [<a href="#references">5</a>]

#### Feature Injection

The first step in Ctrl-X's control mechanism is feature injection from the structure image. After obtaining noised versions of both structure and appearance images via the forward diffusion process at timestep t, these images are passed through the base diffusion model to obtain their intermediate features. For the structure image, the outputs of each convolution layer are captured and directly injected into the corresponding positions in the output image generation process.

![FeatureInjection]({{ '/assets/images/22/FeatureInjection.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Ctrl-X Feature Injection, taken from Ctrl-X video presentation by Jordan Lin* [<a href="#references">8</a>]

This direct feature injection preserves structural information, but unlike ControlNet, it happens without additional network overhead since we're using features from the base model itself. The injection serves as a foundation for maintaining structural alignment while allowing appearance modification through subsequent steps.

One reason why this works is that even in early diffusion timesteps with very noisy images the features already carry some structural information.

![CtrlXEarlyDiffusionFeatures]({{ '/assets/images/22/CtrlXEarlyDiffusionFeatures.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*The authors of the paper visualize the early diffusion features using real, generated, and condition images and feed the noised image through the model at high timesteps (early when sampling) and extract the features of the SDXL model and visualize the top 3 principal components. This illustrates how even at early sampling stages, the structual features already hold decent structural information which allows for semantic correspondence between two images via self-attention* [<a href="#references">5</a>].

#### Spatially Aware Appearance Transfer

![SpatiallyAwareAppearanceTransfer]({{ '/assets/images/22/SpatiallyAwareAppearanceTransfer.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Ctrl-X Spatially Aware Appearance Transfer, taken from Ctrl-X video presentation by Jordan Lin* [<a href="#references">8</a>]

Then to perform spatially-aware appearance transfer:

For some feature layer l:

1. Output features and appearance features are first normalized across their spatial dimensions to isolate structural correspondence from appearance statistics
2. These normalized features are projected through a linear layer to obtain queries from the output features and keys from the appearance features
3. Attention scores are computed between these queries and keys, representing semantic similarity between spatial locations

The resulting attention matrix encodes how different parts of the output image should sample appearance information from the appearance image. For example, if generating a bedroom from a layout, a bed region in the structure would attend strongly to bed-like regions in the appearance image, even if they're in different positions or orientations.

![SelfAttentionSemanticCorrespondence]({{ '/assets/images/22/SelfAttentionSemanticCorrespondence.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Example semantic correspondence extracted from cross-image attention* [<a href="#references">7</a>]

This semantic correspondence is then used to transfer appearance statistics. The attention weights determine how to combine appearance feature statistics spatially:
- A mean feature map is computed as a weighted average of appearance features
- A standard deviation map captures the weighted variance of appearance features
- The output features are normalized and then transformed using these statistics

This approach allows for spatially-varying appearance transfer that respects semantic relationships between structure and appearance images. The normalization step allows the output features to have take the mean and standard deviation of the weighted appearance features. 

#### Self-Attention Injection

After appearance transfer, Ctrl-X performs an additional self-attention injection step to reinforce structural alignment. This step:

1. Takes the attention maps (queries and keys) from the structure image's self-attention operation
2. Uses these attention patterns with values computed from the current output features
3. Applies this modified self-attention result to further align the output with the structural input

![SelfAttentionInjection]({{ '/assets/images/22/SelfAttentionInjection.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*Ctrl-X Self-Attention Injection, taken from Ctrl-X video presentation by Jordan Lin* [<a href="#references">8</a>]

This final step helps maintain structural coherence after appearance transfer by leveraging the base model's learned understanding of spatial relationships. Notably, the self-attention mechanism in the plain SD model is intended for enforcing self-consistency, but since this step is injecting the attention maps from the structural image, this step is enforcing structural consistency between the output image and the structural image rather than self-coherence.

The complete pipeline integrates these steps at each diffusion timestep, allowing the model to maintain both structural and appearance control throughout the generation process. Importantly, this is achieved without any additional training or optimization, making it immediately applicable to any pretrained diffusion model while being computationally efficient compared to methods like ControlNet. Also because these steps are modular, which layers these steps are applied to are hyperparameters that can be tuned.

---

# References 

[1] "What are Diffusion Models?" by Lilian Weng <a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank">https://lilianweng.github.io/posts/2021-07-11-diffusion-models/</a>

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. ArXiv. <a href="https://arxiv.org/abs/1505.04597" target="_blank">https://arxiv.org/abs/1505.04597</a>

[3] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2021). *High-Resolution Image Synthesis with Latent Diffusion Models*. ArXiv. <a href="https://arxiv.org/abs/2112.10752" target="_blank">https://arxiv.org/abs/2112.10752</a>

[4] Zhang, L., Rao, A., & Agrawala, M. (2023). *Adding Conditional Control to Text-to-Image Diffusion Models*. ArXiv. <a href="https://arxiv.org/abs/2302.05543" target="_blank">https://arxiv.org/abs/2302.05543</a>

[5] Lin, K. H., Mo, S., Klingher, B., Mu, F., & Zhou, B. (2024). *Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation Without Guidance*. ArXiv. <a href="https://arxiv.org/abs/2406.07540" target="_blank">https://arxiv.org/abs/2406.07540</a>

[6] Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ArXiv. <a href="https://arxiv.org/abs/1703.10593" target="_blank">https://arxiv.org/abs/1703.10593</a>

[7] Alaluf, Y., Garibi, D., & Patashnik, O. (2023). *Cross-Image Attention for Zero-Shot Appearance Transfer*. ArXiv. <a href="https://arxiv.org/abs/2311.03335" target="_blank">https://arxiv.org/abs/2311.03335</a>

[8] "CS163 - Ctrl-X: Controlling Structure and Appearance for Text-To-Image Generation - Jordan Lin" <a href="https://www.youtube.com/watch?v=ycxkdJnaiuQ" target="_blank">https://www.youtube.com/watch?v=ycxkdJnaiuQ</a>

[9] Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator Architecture for Generative Adversarial Networks*. <a href="https://arxiv.org/abs/1812.04948" target="_blank">https://arxiv.org/abs/1812.04948</a>

---

# Implementation

In this <a href="https://drive.google.com/file/d/1O5pwErojNv6SCF4rVleA7H4mB74eCRLF/view?usp=sharing" target="_blank">Python notebook</a>, we offer a custom implmentation of Image2Image style transfer using ControlNet on the base Stable Diffusion model. We use the HuggingFace Diffusers library to facilitate the construction of an I2I pipeline.

From the HuggingFace Diffusers library, we import the `StableDiffusionControlNetImg2ImgPipeline`, `ControlNetModel`, and `UniPCMultistepScheduler`. The StableDiffusion version is `v1.5`, and the ControlNetModel version is `v11f1p`. We quantize to 16-bit precision and use safetensors for the ControlNetModel. We also set the noise scheduler to UniPCMultistepScheduler to decrease memory utilization while maintaining speed. We use the StableDiffusion base model and the ControlNetModel both quantized to 16-bit floating point precision.

```python
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
```

We use the depth-estimation pipeline from HuggingFace's Transformers library to generate a depth map. This is done by passing the image through the depth-estimation pipeline using the `depth-anything` model and extracting the predicted depth map. We then convert the depth map to a tensor and normalize it to the range $$[0, 1]$$. This depth map is used to condition the ControlNetModel.

```python
import torch
import numpy as np

from transformers import pipeline
from diffusers.utils import load_image, make_image_grid

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
)

depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
depth_map = depth_estimator("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg")["predicted_depth"]
print(depth_map.shape)
depth_map = np.array(depth_map)
depth_map = depth_map[:, :, None]
depth_map = np.concatenate([depth_map, depth_map, depth_map], axis=2)
depth_map = torch.from_numpy(depth_map).float() / 255.0
depth_map = depth_map.permute(2, 0, 1)
depth_map = depth_map.unsqueeze(0).half().to("cuda")
```

We pass the image and depth map through the pipeline. The image is the original image, and the depth map is the depth map estimation generated by the depth-anything model. We also pass in the guiding text prompt "lego harry potter and ron". The results are pretty good, but blur and errors may be attributed to quantization to fp16, the fast noise scheduler for memory optimization, poor depth estimateion, the text prompt, or the base model's generative quality.
```python
output = pipe(
    "lego harry potter and ron", image=image, control_image=depth_map,
).images[0]
make_image_grid([image, output], rows=1, cols=2)
```

We pass the image and depth map through the pipeline. We also pass in the guiding text prompt "lego harry potter and ron". The resulting image is shown below. The left image is the original image, and the right image is the generated image.

The results are pretty good, but blur and errors may be attributed to quantization to fp16, the fast noise scheduler for memory optimization, poor depth estimation, the text prompt, or the base model's generative quality. 

![StableDiffusionControlNetI2I]({{ '/assets/images/22/Implementation.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

*ControlNet Image-to-Image Style Transfer using Stable Diffusion v1.5*

---