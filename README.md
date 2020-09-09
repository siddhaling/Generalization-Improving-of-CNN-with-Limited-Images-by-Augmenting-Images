# Generalization-Improving-of-CNN-with-Limited-Images-by-Augmenting-Images
Use the image augmenting and enhance the number of images to improve the generalization of CNN.

***********************************************************************************************************************
This code demonstrate the use of image augmentation using ImageDataGenerator of Keras to improve generalization
***********************************************************************************************************************

Package Version\
python 3.6.8\
pandas 0.24.1\
Keras 2.2.4\
skimage 0.16.2

One of the most challenge of CNN training is improving the generalization.\
The generalization can improved by increasing the number of image samples.\
However many times, we have limited number of images. One way to increase number\
of samples is using the image augmentation.\

The ImageDataGenerator package provided by Keras can be used to perform the centering, normalization and rotation of images.\
Thus increasing the number of images available for training the model.\

The input to this code is the path to a directory containing images.\
The code will demonstrate the application of ImageDataGenerator on sentiment images.\

### How to run this python code?

To run the CNNimproveGeneralization.py, please provide the path to the director containing images to imFolder.\
As an example, a few images are kept in the folder 'images'.\

Few samples are kept here for complete dataset please refer\
https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

Also it is required to change to current directory using os.chdir().\
The ground truth i.e class label information is to be kept in the file groundTruthMv_a.txt.

Initially the class label information is read and stored in a dataframe.\
All the paths of the images from the given directory are read and a list of paths to images is prepared into allImsPaths.\
This list is then refined to remove the paths which are not part of ground truth file.

A directory is created by storing images content and class label of the image. The prepared directory is called as imDbDict.\
The image data are loaded into variable for the further processing.\
The ImageDataGenerator object is created with featurewise_center=True, featurewise_std_normalization=True and rotation_range=90\
The featurewise_center scales all the images to have mean of pixel values to zero.\
The featurewise_std_normalization makes all the images to have mean of pixel values zero and unit variance.\
rotation_range=90 will randomly perform the rotation of image in range of 0 to 90 degree.\
The train and test split of the image data set is made using skf.split().\
The training data set of images is then loaded into appropriate variables.\
The fit() function is applied on the training image dataset. Using the flow() function the bactches of images are prepared by applying transformations.\
The model is fit using fit_generator() function on the augmented data produced by flow().\
The model is evaluated using model.evaluate() function to provide the performance score.\

# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passioinate Researcher, Focus on Deep Learning and its applications,\
dr.siddhaling@gmail.com
