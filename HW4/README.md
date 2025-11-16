# Assignment 4 

**Name:** Riu Cherdchusakulchai
**Andrew ID:** rcherdch

---

# 1. 3D Gaussian Splatting

## 1.1 3D Gaussian Rasterization (35 points)

<img src="images/q1_render.gif" width="512">


## 1.2 Training 3D Gaussian Representations (15 points)

Learning rates for each parameters

- Opacity: 0.05
- Scale: 0.005
- Color: 0.0025
- Mean: 0.00016

Iteration: 1000

Mean PSNR: 29.647
Mean SSIM: 0.940

<img src="images/q1_training_progress.gif" width="512">

<img src="images/q1_training_final_renders.gif" width="512">


## 1.3 Extensions **(Choose at least one! More than one is extra credit)**

### 1.3.1 Rendering Using Spherical Harmonics (10 Points)

Without Spherical Harmonics

<img src="images/q1_render.gif" width="512">


Using Spherical Harmonics

<img src="images/q1_render_sh.gif" width="512">

**Frame 3**

Without Spherical Harmonics

<img src="images/003.png" width="512">

Using Spherical Harmonics

<img src="images/003_sh.png" width="512">


**Frame 13**

Without Spherical Harmonics

<img src="images/013.png" width="512">

Using Spherical Harmonics

<img src="images/013_sh.png" width="512">

Incorporating spherical harmonics allows Gaussian splatting to model view-dependent effects, resulting in greater fidelity for complex lighting and specular highlights. These benefits are particularly prominent in areas with detailed shading, such as the cushions and armchairs.

### 1.3.2 Training On a Harder Scene (10 Points)

#### Baseline

Learning rates for each parameters

- Opacity: 0.05
- Scale: 0.005
- Color: 0.0025
- Mean: 0.00016
- Spherical Harmonics: 0.000125

Iteration: 1000

Mean PSNR: 17.227
Mean SSIM: 0.642

<img src="images/output_baseline/q1_harder_training_progress.gif" width="512">

<img src="images/output_baseline/q1_harder_training_final_renders.gif" width="512">

#### Improvement

Learning rates for each parameters

- Opacity: 0.05
- Scale: 0.005
- Color: 0.0025
- Mean: 0.00016
- Spherical Harmonics: 0.000125
- Quaternion: 0.001

Iteration: 10000

SSIM loss is added with a weight of 0.2 and isotropic is set to False


Mean PSNR: 20.512
Mean SSIM: 0.731


<img src="images/output_hard/q1_harder_training_progress.gif" width="512">

<img src="images/output_hard/q1_harder_training_final_renders.gif" width="512">


# 2. Diffusion-guided Optimization


## 2.1 SDS Loss + Image Optimization (20 points)

Prompt: "A standing corgi dog", trained for 2000 iterations

No guidance

<img src="images/output4_.png" width="512">

With guidance

<img src="images/output4.png" width="512">

---

Prompt: "A hamburger", trained for 2000 iterations

No guidance

<img src="images/output2_.png" width="512">

With guidance

<img src="images/output2.png" width="512">

---

Prompt: "A gorilla wearing suit with sunglasses in minecraft theme", trained for 2000 iterations

No guidance

<img src="images/output1_.png" width="512">

With guidance

<img src="images/output1.png" width="512">

---

Prompt: "A orange shubby cat with a black strip holding sword", trained for 2000 iterations

No guidance

<img src="images/output3_.png" width="512">

With guidance

<img src="images/output3.png" width="512">

## 2.2 Texture Map Optimization for Mesh (15 points)

Prompt: A pink and yellow stripe cow

<img src="images/py_cow.gif" width="512">

Prompt: A black and white cow


<img src="images/bw_cow.gif" width="512">


## 2.3 NeRF Optimization (15 points)

A standing corgi dog


<img src="images/c0/rgb_ep_99.gif" width="512">

<img src="images/c0/depth_ep_99.gif" width="512">

A hamburger

<img src="images/h0/rgb_ep_99.gif" width="512">

<img src="images/h0/depth_ep_99.gif" width="512">


A hotdog

<img src="images/hd0/rgb_ep_99.gif" width="512">

<img src="images/hd0/depth_ep_99.gif" width="512">

## 2.4 Extensions (Choose at least one! More than one is extra credit)

### 2.4.1 View-dependent text embedding (10 points)

A standing corgi dog

Without view dependence

<img src="images/c0/rgb_ep_99.gif" width="512">

<img src="images/c0/depth_ep_99.gif" width="512">


Using view dependence


<img src="images/c1/rgb_ep_99.gif" width="512">

<img src="images/c1/depth_ep_99.gif" width="512">


A hamburger

Without view dependence

<img src="images/h0/rgb_ep_99.gif" width="512">

<img src="images/h0/depth_ep_99.gif" width="512">

Using view dependence

<img src="images/h1/rgb_ep_99.gif" width="512">

<img src="images/h1/depth_ep_99.gif" width="512">

A hotdog

Without view dependence

<img src="images/hd0/rgb_ep_99.gif" width="512">

<img src="images/hd0/depth_ep_99.gif" width="512">

Using view dependence

<img src="images/hd1/rgb_ep_99.gif" width="512">

<img src="images/hd1/depth_ep_99.gif" width="512">


Previously, using just one generic prompt confused the model, causing the Janus problem where the detailed front view would weirdly repeat on the other side (like the corgi ears). View-dependent conditioning fixes this by explicitly telling the model what the sides and back should look like. 


