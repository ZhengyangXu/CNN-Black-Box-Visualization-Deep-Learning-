# GR5242_Project-3-Feature-visualization-2018fall
This is a brieft introduction of our Advanced machine learning 2018 fall final project. For details and results, please read the report of our project.
## 1. Introduction 

Neural networks are usually considered as black boxes, due to its complexity. For a neural network with multiple layers, it is difficult to figure out the features learnt by those hidden layers. In this project, we try to visualize the features learned by the filters of each convolutional layers of a CNN.  

### 1.1	Data Used

In this project, MNIST dataset, SVHN dataset and CIFAR10 dataset were used.  

The MNIST dataset of handwriting digit images contains a training set with 60000 images, and a test set with 10000 images. Each input is a 28*28 black and white image. Each output is a label with 10 classes. Digit 1 has label 1, and digit 9 has label 9.

For the SVHN, we use the cropped digits. In the dataset, each input is a 32*32 RGB image. The outputs are labels. There are 10 classes, and one for each digit. 0 has label 10. 1 has label 1, and 9 has label 9. The training set contains 73257 digits, and the test set has 26032 digits.  

The CIFAR-10 dataset has 60000 32x32 RGB images in 10 classes. Each class contains 6000 images. The training set contains 50000 images, and the test set has 10000 images.  
 
### 1.2	Framework of Project

In this project, we used keras. We started from training a simple 2-layers CNN on MNIST. Since the project mainly focused on feature visualization, when building the convnet, we used the code provided by Eijaz Allibhai, in the post Building a Convolutional Neural Network (CNN) in Keras. After that, we first visualized the activation maps given by each filter in the two convolutional layers. Then, through gradient ascent, we found an image for each filter that maximally activated the filter. To make the resulted images more interpretable, we tried to apply some kinds regularization.

Later, we turned to SVHN dataset. Again, we trained a simple CNN on SVHN dataset, and tested whether the steps mentioned above can be applied. 

For our curiosity, we also tried to apply the methods to the CNN trained on CIFAR10 dataset. We wished to know what would learnt by the each layer’s filters, in a CNN trained on a dataset containing more complicated images.
  
### 1.3	Preparation

Since feature visualization was not covered in class, we did some researches on it. 
We watched the video of a Stamford open course, named Visualizing and Understanding. Through the video, we got some a rough idea of feature visualization. Then we read the two Google posts and several related papers, like Visualizing and Understanding Convolutional Networks and Understanding Neural Networks Through Deep Visualization. After finishing the reading, we discussed a few times and figured out the framework of our project.

### 1.4 Group members

+ Chenghao Yu cy2475@columbia.edu	Sec 002

+ Zhengyang Xu zx2229@columbia.eu	Sec 001

+ Yingqiao Zhang yz3209@columiba.edu	Sec 001

+ Yanxin Li yl3774@columbia.edu	Sec 001

### 1.5 Structure

(1) doc -- GR5242 final report.pdf -- final report in pdf
           We seperate our ipynb and put them into two folders, cifar10 and mnist&svhn

(2) data -- this folder contains CIFAR10 dataset (Need to download again since it is too big to upload)

(3) output -- this folder contains saved trained models and some intermediate output (Need to generate again since too big to upload)

(4) lib -- this folder contains function we used

### 1.6 Rerun code

+ To rerun the code, please go to the doc folder, and open the ipynb files to run the code, and there are two ipynb files. Please notify that since the Github upload limitation and different working directory, you may not run it smoothly. Please download the relative data and change your working directory to rerun the code.

### 1.7 Reference (Just part, please find all of them in the final report)

+ https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
+ https://www.manning.com/books/deep-learning-with-python
+ [Visualizing Higher-Layer Features of a Deep Network 1 Introduction](https://pdfs.semanticscholar.org/65d9/94fb778a8d9e0f632659fb33a082949a50d3.pdf)
+ http://ufldl.stanford.edu/housenumbers/
+ https://distill.pub/2017/feature-visualization/

