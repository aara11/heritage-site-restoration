# heritage-site-restoration
This project is implemented towards my thesis.

### Introduction

This repo contains the code implemented for thesis. The aim of this project is to develop a deep neural network based domain specific image
reconstruction engine for heritage site restoration. The following objectives have been
set to achieve this aim:
- Learning the manifolds of an image to obtain latent space representation and generating the original structure of the input image.
- Reconstruction of the generated image for fine refining of the captured features in order to obtain a structurally coherent version of the original image.

### Dataset
Bishnupur Heritage Image Dataset[1] is used for this project. This Dataset consists of high definition photos of different temples. The walls of the temples are decorated with terracotta panels which present floral and geometric patterns in addition to various scenes from mythological stories.

### Dataset Preparation
For obtaining the two dimensional frames, warp perspective transform[2] is performed. Manually selected portion of the wall to obtain 3×3 affine transform matrix T which is then used to compute the geometrically correct version. This warped image is further segmented to obtain individual square panels of 64×64 pixels training and testing purpose.
Script: data_generate.py

#### Gray scale conversion
They are converted to grayscale by histogram equalization.
Script: data_binarization_resize.py

Split the dataset into Training, Validation and Testing datasets.
Script: split_dateset_train_validation.py

### Methodology
#### Deep Generative Adversarial Autoencoder
Adversarial Autoencoder [3] are a type of GANs where the encoder is trained to produce a latent encoding with aggregated posterior distribution q(z) similar to a target prior distribution p(z). The generator is trained to produce realistic images for the given input prior distribution 
![Adversarial AutoEncoder](https://github.com/aara11/heritage-site-restoration/blob/main/doc/aae.png?raw=true)

The input to the encoder is an image of shape 64 × 64. Each down-conv block is followed by Batch normalization with ε = 10−05 and momentum = 0.1, and ReLU.

The bottleneck layer encodes the manifolds for the input image to give 100 dimensional latent space representation with normal posterior distribution of N(0,1).
![Adversarial AutoEncoder Architecture](https://github.com/aara11/heritage-site-restoration/blob/main/doc/encoder-decoder.png?raw=true)


Scripts:
Train the model - train_aae.py
Plot Loss -  plot_logger_loss.py
Generate Synthetic Images for random encoding - generate_synthetic_images.py
Create a 3D plot - plot_3d.py

#### Pixel Level Reconstructon Engine
Script:
Create dataset for reconstruction engine - create_dataset_refiner.py
Train pixel-level refiner - train_pixel_rnn.py


### Results
Scripts:
compare_cosine_similarity_vgg penultimate.py'
compare_SSIM_scores.py


#### References
[1] https://www.isical.ac.in/bsnpr
[2] Wolberg, G. (1990). Digital image warping, Vol. 10662, IEEE computer society press
Los Alamitos, CA.
[3] Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I. and Frey, B. (2015). Adversarial
autoencoders, arXiv preprint arXiv:1511.05644

