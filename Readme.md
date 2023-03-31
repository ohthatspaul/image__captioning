# Image Captioning

# Dataset

> ### Images : [Flickr8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
>
> ### Text : [Flickr8k_text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
>
> <br>

# Data preprocessing and feature extraction

> <br>

# Built and trained Encoder-Decoder Model

> <br>

# Model Evaluation

> The features of the images to be captioned are extracted using a VGG16 CNN that as been pretrained using ImageNet data. The output of the FC8 layer of VGG16 is fed to another dense layer to extract the image feature. The weights of the VGG16 CNN are frozen when we train our RNN language generator. The additional dense layer is trainable
> The words are embedded using the pretrained wording GloVE embeddings. The embeddings are not trained further in our generative model.
> The image features and word embeddings are concatenated together and input to an LSTM RNN. Te output of the RNN is sampled as used as caption to the image. A special "START" token is used to initialize the generation process, and the process is terminated when an "END" token is sampled.

> <br>

# Hosting and Deployment
