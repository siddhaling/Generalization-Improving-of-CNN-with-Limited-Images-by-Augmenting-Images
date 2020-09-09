import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import gray2rgb
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import sgd
from keras import backend as K
from gabor_init_withSelectedParams import gabor_init_withSelectedParams
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

#change to current directory
os.chdir('')
#folder containing images
imFolder='images'
#ground truth file
grndTruthPath='groundTruthLabel.txt'

#change dimension of the image
dim=(256,256)
input_shape = (dim[0],dim[1],3)
num_classes = 2

#collect all the images present in the folder
dirs=os.listdir(imFolder)
#column name to be read from ground truth file
clNames=['Image','Sentiment']
clsLabels=pd.read_csv(grndTruthPath,names=clNames,delimiter='\t')
clsLabels.set_index('Image',inplace=True)

#Read the file name from the given image
#check the corresponding class present in the ground truth file by looking into dataframe clsLabels
#in this example there are two classes positive and negative, the positive class is set to 1 and negative is set to 0
#read the images from the folder perform preprocessing such as dimension change, gray2rgb
#store the both image and class into the dictonary imDbDict
def createImagesSet(allImagesFoldrPath,dim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),dim[0],dim[1],3))
    imDbDict={}
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        print(allImagesFoldrPath[im])
        imNamge=allImagesFoldrPath[im].split('\\')[-1]
        actualClass=clsLabels.loc[imNamge,'Sentiment']
        
        if (actualClass=='positive'):
            y_Set[im]=1
        else:
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,dim)
        x_imageSet[im]=readImage
        imDbDict[allImagesFoldrPath[im]]=(x_imageSet[im],y_Set[im])
    return imDbDict

#Check whether the given image is part of the ground truth file
#collect all the names of the images into two category images present in ground truth and another not present in ground truth
def collectImNames(entireDb):
    imNmPresentInGrndTrth=[]
    imPathNotPresentInGrndTrth=[]
    for imPath in range(len(entireDb)):
        imNm=entireDb[imPath].split('\\')[-1]
        if imNm in clsLabels.index:
            imNmPresentInGrndTrth.append(imNm)
        else:
            imPathNotPresentInGrndTrth.append(entireDb[imPath])
    return imNmPresentInGrndTrth,imPathNotPresentInGrndTrth

#Prepare the train and test set based on the path for training samples and test samples are received.
#It utilizes keras tool to_categorical to convert numerical class into one hot representation
def load_data(allImagesTrainPath,allImagesTestPath,imDbDict):
    x_trainImSet=np.empty((len(allImagesTrainPath),dim[0],dim[1],3))
    x_testImSet=np.empty((len(allImagesTestPath),dim[0],dim[1],3))
    y_trainSet=np.zeros(len(allImagesTrainPath))
    y_testSet=np.zeros(len(allImagesTestPath))
    for trnPi in range(len(allImagesTrainPath)):
        (x_trainImSet[trnPi],y_trainSet[trnPi])=imDbDict[allImagesTrainPath[trnPi]]
    
    for testPi in range(len(allImagesTestPath)):
        (x_testImSet[testPi],y_testSet[testPi])=imDbDict[allImagesTestPath[testPi]]
        
    x_trainImSet= x_trainImSet.astype('float32')
    x_testImSet= x_testImSet.astype('float32')
    y_trainSetBinary=y_trainSet
    y_testSetBinary=y_testSet
# convert class vectors to matrices as binary
    y_trainSet= keras.utils.to_categorical(y_trainSet, num_classes)
    y_testSet= keras.utils.to_categorical(y_testSet, num_classes)
    
    print('Number of train samples in Dataset: ', x_trainImSet.shape[0])
    print('Number of test samples in Dataset: ', y_testSet.shape[0])
    
    return (x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)

# Build Convolutional Neural Network Model
def build_model(num_filters_input_layer):
    model = Sequential()
    
    # Convolution layer 1
    model.add(Conv2D(num_filters_input_layer, kernel_size=(11,11), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros', 
                     input_shape=input_shape))
    model.add(Activation('relu'))
                  
    # Convolution layer 2
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
        
    model.add(MaxPooling2D(pool_size=(2,2)))
              
    # Convolution layer 3
    model.add(Conv2D(192, kernel_size=(5,5), padding='same', 
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
                  
    # Convolution layer 4
    model.add(Conv2D(192, kernel_size=(5,5), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
        
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # Convolution layer 5
    model.add(Conv2D(96, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
        
    model.add(Conv2D(96, kernel_size=(3,3), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
        
    model.add(Conv2D(48, kernel_size=(1,1), padding='same',
                     kernel_initializer='glorot_uniform', 
                     bias_initializer='zeros'))
    model.add(Activation('relu'))
        
    model.add(AveragePooling2D(pool_size=(2,2)))
              
    model.add(Flatten())
    
    model.add(Dense(num_classes, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))
    
    return model

#Parameter selection for CNN model
numOfInputFilters=[8,10]
batch_size = 10
epochs = 25
seed = 11
BatchSize=10

#Data frame to store performance measurements
performanceTb=pd.DataFrame(columns=['FilterNum','FoldNum','TrnOrTest','TN', 'FP', 'FN', 'TP'])

#collect all the path of images
allImsPaths=[(imPath+di) for di in dirs if('txt' not in di)]
#remove images not present in ground truth table
imNmPresentInGrndTrth,imPathNotPresentInGrndTrth=collectImNames(allImsPaths)
labels=list(clsLabels.loc[imNmPresentInGrndTrth,'Sentiment'])
for rPath in imPathNotPresentInGrndTrth:
    allImsPaths.remove(rPath)

#create an image data set
imDbDict=createImagesSet(allImsPaths,dim,clsLabels)
skf = StratifiedKFold(n_splits=5,shuffle=True, random_state=seed)

#Augment the images using image data generator function with image centering, normalization and rotation. 
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,rotation_range=90)
Total_score=0
for fil in numOfInputFilters:   
    print('Working for the model(%s)'%fil)
#initialize and create the CNN model
    model = build_model(fil)
#Set the optimizer for CNN model
    optimizer = sgd(0.01, 0.9, 0.0005, nesterov=True)
#compile the CNN model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#perform the Cross validation on CNN model
    foldi=0
    for trn_indx, test_indx in skf.split(allImsPaths, labels):
        print("trn_indx: %s test_indx: %s" % (len(trn_indx), len(test_indx)))    
        trainSetImagesPath=[allImsPaths[indx] for indx in trn_indx]
        testSetImagesPath=[allImsPaths[indx] for indx in test_indx]
        (x_trainImSet,y_trainSet,y_trainSetBinary), (x_testImSet,y_testSet,y_testSetBinary)=load_data(trainSetImagesPath,testSetImagesPath,imDbDict)
#generate the augmented image data on selected fold of the training set.
        train_datagen.fit(x_trainImSet)
        train_generator=train_datagen.flow(x_trainImSet, y_trainSet, batch_size=BatchSize)
        model.fit_generator(train_generator,steps_per_epoch=len(x_trainImSet)/BatchSize,validation_data=(x_testImSet,y_testSet),epochs=epochs,shuffle=True)
#Evaluate on the trained model.
		score = model.evaluate(x_testImSet, y_testSet)
		Total_score+=score
		foldi=foldi+1
    	print("Cross validation Accuracy=",Total_score/foldi)