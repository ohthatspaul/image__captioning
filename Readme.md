# Image Captioning

## Dataset

---

## Images : [Flickr8k_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

## Text : [Flickr8k_text](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)

## Data preprocessing and feature extraction

---

When using the RNN as the languate model and a affine network to generate words, we need to feed the already generated caption into the model and get the next word. Therefore, to generate a caption of n words, the mode needs to run n+1 times (n words plus token). During training, we also need to run the model n+1 times, and generate a separate training sequence for each run. There are 6000 images in the training data set, and 5 captions for each image. The maximum length of the caption is 33 words. This comes to a maximum of <code> 6000×5×33 or 990,000 </code> training samples. The training data is generated on-the-fly, just before the model requires it. That is, we will generate the training data one batch at a time, and then input the data into the model as needed (Progressive Loading).

We first define a module that a caption of <code>length n and generates the n+1 </code>training data.

## Building and trainining Encoder-Decoder Model

---

Made use of the VGG16 model as the base model for the CNN. We replace the last softmax layer freeze with another affine layer with 256 output and add a dropout layer. The original layers of the VGG16 model is frozen. The image is input into the input of the VGG16 layer. The GLOVE embedding parameters are also frozen. Words are fed as input to the embedding. The output of the embedding is fed into an LSTM RNN with 256 states. The output of the LSTM (256 dimensionss) and the output of the CNN (256 dimensions) is concatenated together to for a 512 dimensional input to a dense layer. The output of the dense layer is fed into a softmax function.

> ![model_concat image](https://github.com/ohthatspaul/image__captioning/blob/main/model_concat.png)

At each time step of the RNN the a word in the training caption will be fed into the input of the RNN. The out of the RNN will be compared with the ground truth next word. The training captions were tokenized and embedded using the GLOVE word embeddings. The embeddings were fed into the RNN.

 <br>

## Model Evaluation

---

The model outputs a one-hot vector and the error was measured using categorical crossentropy.Generating novel image captions using the trained model. Test images and images from the internet were used as input to the trained model to generate captions. The captions were examined to determine the weaknesses of the model and suggest improvements. Beam search was used to generate better captions using the model.The model was evaluated using the BLEU and ROUGE metric. Used a RNN as a language model. With this approach, the word embeddings are input to the RNN, and the final state of the RNN is combined with image features and input to another neural network to predict the next word in the caption. The features of the images to be captioned are extracted using a VGG16 CNN that as been pretrained using ImageNet data. The output of the <code> FC8 layer of VGG16 </code> is used as the image feature. The weights of the VGG16 CNN are frozen when we train our RNN language generator. The additional dense layer is trainable. The words are embedded using the pretrained wording GloVE embeddings. The embeddings are not trained further in our generative model. The image features and word embeddings are concatenated together and input to an LSTM RNN. The output of the RNN is sampled as used as caption to the image. A special "START" token is used to initialize the generation process, and the process is terminated when an "END" token is sampled.

## Hosting and Deployment
