Generative Adversarial Networks implementation on MNIST
=============

files/directories:
GAN.py : GAN Implementation
data   : MNIST images seperated into classes
images : Network generated images
models : pretrained model

## Dataset

These models were trained using MNIST handwritten dataset.

The training dataset consists of 7,781 28x28 gray-scale images and these images will look like these (sample: 1, 4)
![Alt text](images/train1.jpg?raw=true "Training image sample from class 1")
![Alt text](images/train2.jpg?raw=true "Training image sample from class 4")

## Generative adversarial networks
Generative adversarial networks are a type of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks competing against each other in a zero-sum game framework. One network is generative and one is discriminative.[1] Typically, the generative network is taught to map from a latent space to a particular data distribution of interest, and the discriminative network is simultaneously taught to discriminate between instances from the true data distribution and synthesized instances produced by the generator. The generative network's training objective is to increase the error rate of the discriminative network (i.e., "fool" the discriminator network by producing novel synthesized instances that appear to have come from the true data distribution). These models are used for computer vision tasks

## Training
To train GAN we need two network architectures (discriminator & generator). discriminator gets images from either our real dataset or a from a generator which uses normally distributed noise as input and churns out an image of size 28x28. The discriminator`s job is output the probablity if the input image it got was real or fake.
Thus we optimize our loss function for discriminator & generator in these ways. g_loss is the generator loss for when discriminator correctly classifies fake images as fake. Thus we encourage generator to produce more ralistic images. d_loss_real & d_loss_fake are discriminator losses encouraging it to correctly classify ie. real images as real, fake as fake.

```python
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg))) 
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))#prob of being fake so optimize to 0
``` 
When the network converges generator becomes so good at generating images that the discriminator starts failing to distinguish real and fake images.

## Testing

Images can be viewd on tensorboard. Here are few generated images.
![Alt text](images/gen1.png?raw=true "generated image 1")
![Alt text](images/gen2.png?raw=true "generated image 2")
![Alt text](images/gen3.png?raw=true "generated image 3")
![Alt text](images/gen4.png?raw=true "generated image 4")
![Alt text](images/gen5.png?raw=true "generated image 5")
![Alt text](images/gen6.png?raw=true "generated image 6")
![Alt text](images/gen7.png?raw=true "generated image 7")
![Alt text](images/gen8.png?raw=true "generated image 8")

## Additional References
1. https://arxiv.org/abs/1406.2661
2. https://github.com/llSourcell/Generative_Adversarial_networks_LIVE/blob/master/EZGAN.ipynb