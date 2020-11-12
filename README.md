# Explainable-AI

<br>
<br>

## What is Explainable AI?
Explainable AI are the set of methods and techniques in the application of artificial
intelligence such that the result of the solution can be be understood by humans. It
helps us to solve the problem of Black-Box in machine learning where it is very
difficult to explain how a model has approached some specific decision.

<br>

## Why Explainable AI?
Sometimes its not enough to just make blind predictions. You need to have some
justification regarding why the model is predicting some specific decision. For
example:
1. If a self-driving car makes a bad decision and kills a person, if we aren’t able to
quantify the reason for this bad decision, then we won’t be able to rectify it, which
could lead to even more disasters.

<br>

2. If some image recognition system is trained to detect a tumour in images and
performs very well in terms of accuracy both on validation and test set. But when you
present the results to the stakeholders they question from what parts of the image
your model is learning or what is the main cause of this output and your most
probable answer would be “I don’t know” and no matter how perfect your model is,
the stakeholders won’t accept it because human life is at stake.
With the increased research in the field of Machine Learning especially Deep
Learning various efforts are being made to solve the problem of interpretability and
reach the stage of interpretable AI.

<br>

Although there are many interpretation methods, we will focus on 2 widely used methods namely:

<br>

## 1. Saliency Maps:
So the idea is pretty straightforward, We compute the gradient of output category with respect to input image. This should tell us how output category value changeswith respect to a small change in input image pixels. All the positive values in the gradients tell us that a small change to that pixel will increase the output value.

<br>

<img src="https://github.com/HimanchalChandra/Explainable-AI/blob/main/saliency_map.png" alt="Image4" width="700" height="300"/>     

## 2. Gradient Class Activation Map (GRADCAM):
Gradient Class Activation Map (Grad-CAM) for a particular category indicates the discriminative image regions used by the CNN to identify that category.
GRAD-CAM utilizes the last convolutional layer feature maps beacause it retain spatial information (which is lost in fully-connected layers) and use it to visualize the decision made by CNN.

<br>

Some feature maps would be more important to make a decision on one class than others, so weights should depend on the class of interest.

<br>

<img src="https://github.com/HimanchalChandra/Explainable-AI/blob/main/grad-cam.png" alt="Image4" width="700" height="300"/>     

## Code: Training Part:
I used Google Colab for the coding part. I downloaded the Dog Vs Cat dataset from
Kaggle and trained it on the VGG16 network (Transfer Learning) keeping the
convolutional layers frozen.

<br>

I used 19000 images for training and 6000 images for testing and achieved almost
100 percent accuracy on the 8th epoch because of the virtue of transfer learning.
I used early stopping to stop training when the monitored metric which was
Validation accuracy has stopped improving, and Model checkpoint to save the best
weights.

<br>

## Code: Visualization Part:

## Saliency Maps:
I used the Keras-vis toolkit for visualizing Saliency maps and
Matplotlib for plotting.

<br>

## GRAD-CAM:
For visualizing grad-cam, I manually calculated the gradients with the
help of Yumi’s tutorial on Grad-Cam.
