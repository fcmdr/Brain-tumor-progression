#Load librairies 
#!pip install nibabel
#import nibabel as nib
#!pip install numpy
#!pip install math
#!pip install glob2
#!pip install scipy.ndimage
#!pip install pandas
#!pip install cv2
import numpy as np
import math
import matplotlib.pyplot as plt
#!python -m pip install pydicom
import pydicom
import random
from pydicom.data import get_testdata_files
import glob
#!pip install pydicom
import pydicom as dicom
#import PIL # optional
import pandas as pd
#import csv
import cv2
import matplotlib.pyplot as plt
import imageio
import os, shutil
import glob2
#import pydicom
import scipy.ndimage as ndi 
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,TensorBoard
from tensorflow.keras import backend as keras
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

def display_image(array):
    """Display an image from probabilities values given by the unet model to complete"""
    #Transform the matrix mask containing probabilities (between 0 and 1) into a binary (O-1) matrix
    results_bin = np.where(np.array(results) > 0.5, 1, 0 )#equals 1 if greater than 0.5
    #Transform pixels values from binary to 0-255
    results_norm = np.asarray(np.where(np.array(results_bin)==1, 255, 0 ))
    #Reshape the dimensions to 256*256 (initially 1,256,256,1)
    results_norm1 = results_norm[0,:, :, 0]
    results = my_beloved_model.predict(X_train)
    #print(results_norm1.shape)
    #save the image gives name "brain_mask_pred.jpg" #Without training !
    io.imsave('D:/unet/brain_mask_pred.jpg',results_norm1)
    
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
        
def getPixelDataFromDicom(filename):
    """Get pixel values from a dicom file""" 
    return pydicom.read_file(filename).pixel_array

def split_dataframe(dataframe,train_prop,one_minus_test_prop):
    """Split randomly the dataframe in which images and masks paths are keeped into train,validation and test"""
    train, validate, test = np.split(dataframe.sample(frac=1,random_state = 2), [int(train_prop*len(dataframe)), int(one_minus_test_prop*len(dataframe))])
    assert len(dataframe) == len(train) + len(validate) + len(test)
    return train, validate, test



def image_gen(dataframe,batch_size):
    """From a dataframe containing path of images and masks and the size of the batch, it creates a generator that yield an image and its corresponding matrix""" 
    # Iterate over all the image paths
    counter = 0
    size_dataframe = len(dataframe) #or shape[0]
    print(size_all_df)
    for i in range(0,math.floor(size_dataframe/float(batch_size)),1):  #Round down to the nearest integer as we want integer batch_size
        #build lst which is a list of list (made of the paths of the image and the corresponding mask)
        lst = list(zip(dataframe['dcm_path'],dataframe['Mask']))[i*batch_size:(i+1)*batch_size]
        # Transform the list that in (1,2) dimension into an numpy array of dimension (1,)
        X = np.array(lst)[:,0] #be careful lst is not a numpy array so we cannot index it like [:,0] for example to take the 1st element (image)
        #Get the pixels information from the dicom files for the images and for the masks
        X = [getPixelDataFromDicom(X_matrix) for X_matrix in X]
        #dimension of X_resized is (1,256,256)
        X_resized = np.array([cv2.resize(x_resized, (256,256), interpolation=cv2.INTER_LINEAR) for x_resized in X])
        #We add a dimension with expand_dims axis=-1 to get (1,256,256,1)
        X_resized = np.expand_dims(X_resized, axis=-1)
        #We apply the same steps for the masks
        Y = np.array(lst)[:,1]
        Y = [getPixelDataFromDicom(Y_matrix) for Y_matrix in Y]
        Y_resized = np.array([cv2.resize(y_resized, (256,256), interpolation=cv2.INTER_LINEAR) for y_resized in Y])
        Y_resized = np.expand_dims(Y_resized, axis=-1)
        yield X_resized,Y_resized

def transform_dcm_im_jpg(folder_list_to_transform,PNG = False):
    PNG = False
    
    for folder_ in folder_list_to_transform:
        #Build derived folder from the initial ones to put the "jpg" files
        jpg_folder_path = os.path.join(folder_ + "_jpg")
        # Specify the output jpg/png folder path
        #Create the folders
        #os.mkdir(jpg_folder_path)
        #List the folders (images and masks)
        images_path = os.listdir(folder_)
        for n, image in enumerate(images_path):
            ds = dicom.dcmread(os.path.join(folder_, image))
            pixel_array_numpy = ds.pixel_array
            if PNG == False:
                image = image.replace('.dcm', '.jpg')
            else:
                image = image.replace('.dcm', '.png')
            cv2.imwrite(os.path.join(jpg_folder_path, image), pixel_array_numpy)
            #if n % 50 == 0:
            #    print('{} image converted'.format(n))
        
def unet(pretrained_weights,input_size = (256,256,1)):
    """Create a unet segmentation model with Keras API loading existing weight (unet_membrane.hdf5), then fixes the input size of the image to (256,256,1)""" 
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = os.listdir(img_folder) #List of training images
    random.shuffle(n)
    while (True):
        img = np.zeros((batch_size, 512, 512, 1)).astype('float')
        mask = np.zeros((batch_size, 512, 512, 1)).astype('float')

        for i in range(c, c+batch_size): #initially from 0 to 16, c = 0. 
            X = np.array(lst)[:,0]
            X = [getPixelDataFromDicom(X_matrix) for X_matrix in X]
            X_resized = np.array([cv2.resize(x_resized, (512,512), interpolation=cv2.INTER_LINEAR) for x_resized in X])
            train_img = np.expand_dims(X_resized, axis=-1)
            Y = np.array(lst)[:,1]
            Y = [getPixelDataFromDicom(Y_matrix) for Y_matrix in Y]
            Y_resized = np.array([cv2.resize(y_resized, (512,512), interpolation=cv2.INTER_LINEAR) for y_resized in Y])
            mask = np.expand_dims(Y_resized, axis=-1)
            

            img[i-c] = train_img #add to array - img[0], img[1], and so on.


            train_mask = cv2.imread(mask_folder+'/'+n[i], cv2.IMREAD_GRAYSCALE)/255.
            train_mask = cv2.resize(train_mask, (512, 512))
            train_mask = train_mask.reshape(512, 512, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

            mask[i-c] = train_mask

            c+=batch_size
            if(c+batch_size>=len(os.listdir(img_folder))):
                c=0
                random.shuffle(n)
                  # print "randomizing again"
            yield img, mask
            
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

#def dice_coefficient_loss(ytrue, ypred):
#    return -dice_coef(y_true, y_pred, smooth=1)

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
        
def new_train_gen(image_gen,mask_gen):
    for element1 in image_gen:
        res1 = element1
    for element2 in mask_gen:
        res2 = element2
        yield res1,res2
    
if __name__ == '__main__':
    b_size = 1 
    loss=[]
    accuracy=[] 
    val_loss=[] 
    val_accuracy=[]
    
    #shuffle the dataset 
    all_df = pd.read_csv(r'D:/unet/all_df_images_masks.csv').sample(frac=1)
    all_df_train,all_df_val,all_df_test = split_dataframe(all_df,0.7,0.9)
    size_all_df = len(all_df)
    size_all_df_train = len(all_df_train)
    size_all_df_val = len(all_df_test)
    
    #Define folders
    fit_gen_brain = 'D:/fit_gen_brain'  
    train_image_brain = os.path.join(fit_gen_brain,'train_images')
    #os.mkdir(train_image_brain)
    validation_image_brain = os.path.join(fit_gen_brain,'val_images')
    #os.mkdir(validation_image_brain)
    test_image_brain = os.path.join(fit_gen_brain,'test_images')
    #os.mkdir(test_image_brain)
    
    train_image_brain2 = os.path.join(fit_gen_brain,'train_images2')
    #os.mkdir(train_image_brain2)
    validation_image_brain2 = os.path.join(fit_gen_brain,'val_images2')
    #os.mkdir(validation_image_brain2)
    test_image_brain2 = os.path.join(fit_gen_brain,'test_images2')
    #os.mkdir(test_image_brain2)
    
    #Create masks folders
    train_masks_brain = os.path.join(fit_gen_brain,'train_masks')
    validation_masks_brain = os.path.join(fit_gen_brain,'val_masks')
    test_masks_brain = os.path.join(fit_gen_brain,'test_masks')

    #Create masks folders with images instead of dicom files
    train_masks_brain2 = os.path.join(fit_gen_brain,'train_masks2')
    #os.mkdir(train_masks_brain2)
    validation_masks_brain2 = os.path.join(fit_gen_brain,'val_masks2')
    #os.mkdir(validation_masks_brain2)
    test_masks_brain2 = os.path.join(fit_gen_brain,'test_masks2')
    #os.mkdir(test_masks_brain2)
    
    #Create jpg folders images
    train_image_brain2 = os.path.join(fit_gen_brain,'train_images2')
    validation_image_brain2 = os.path.join(fit_gen_brain,'val_images2')
    test_image_brain2 = os.path.join(fit_gen_brain,'test_images2')
    
    
    #Copy dicom files in another folder
    #Copy train images into the train_image_brain2 (images instead of dicom files)
    for image_ in os.listdir(train_image_brain):
        shutil.copy(os.path.join(train_image_brain,image_) , os.path.join(train_image_brain2,os.path.basename(image_))) 
    #Copy val images into the val_image_brain2 (images instead of dicom files)
    for image_ in os.listdir(validation_image_brain):
        shutil.copy(os.path.join(validation_image_brain,image_) , os.path.join(validation_image_brain2, os.path.basename(image_))) 
    #Copy test images into the test_image_brain2 (images instead of dicom files)
    for image_ in os.listdir(test_image_brain):
        shutil.copy(os.path.join(test_image_brain,image_) , os.path.join(test_image_brain2,os.path.basename(image_))) 

        #Then masks
        #Copy train images into the train_image_brain2 (images instead of dicom files)
    for image_ in os.listdir(train_masks_brain):
        shutil.copy(os.path.join(train_masks_brain,image_) , os.path.join(train_masks_brain2,os.path.basename(image_))) 
    #Copy val images into the val_image_brain2 (images instead of dicom files)
    for mask in os.listdir(validation_masks_brain):
        shutil.copy(os.path.join(validation_masks_brain,mask) , os.path.join(validation_masks_brain2, os.path.basename(mask))) 
    #Copy test images into the test_image_brain2 (images instead of dicom files)
    for mask in os.listdir(test_masks_brain):
        shutil.copy(os.path.join(test_masks_brain,mask) , os.path.join(test_masks_brain2,os.path.basename(mask))) 

    
    #Convert dcm files into images
    folder_list_to_transform =[train_image_brain2,validation_image_brain2,test_image_brain2,train_masks_brain2,validation_masks_brain2,test_masks_brain2]

    transform_dcm_im_jpg(folder_list_to_transform,PNG = False)
    
    #Instantiate the model and save it in "my_beloved_model"
    my_beloved_model = unet(pretrained_weights=r"D:/unet/unet_membrane.hdf5",input_size = (256,256,1))
    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #compile the model
    my_beloved_model.compile(
              optimizer = opt,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
    

    #create train and validation generator
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
      
    NO_OF_EPOCHS = 4
    BATCH_SIZE = 1
    
    train_image_brain2_jpg = "D:\\fit_gen_brain\\train_images2_jpg"
    validation_image_brain2_jpg = "D:\\fit_gen_brain\\val_images2_jpg"
    test_image_brain2_jpg = "D:\\fit_gen_brain\\test_images2_jpg"
        
    train_masks_brain2_jpg = "D:\\fit_gen_brain\\train_masks2_jpg"
    validation_masks_brain2_jpg = "D:\\fit_gen_brain\\val_masks2_jpg"
    test_masks_brain2_jpg = "D:\\fit_gen_brain\\test_masks2_jpg"
        
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_image_generator = train_datagen.flow_from_directory(train_image_brain2_jpg,batch_size = BATCH_SIZE)
    train_mask_generator = train_datagen.flow_from_directory(train_masks_brain2_jpg,batch_size = BATCH_SIZE)
    val_image_generator = val_datagen.flow_from_directory(validation_image_brain2_jpg, batch_size = BATCH_SIZE)
    val_mask_generator = val_datagen.flow_from_directory(validation_masks_brain2_jpg,batch_size = BATCH_SIZE)
    
    #Build a couple of image and its corresponding mask
    #train_generator = zip(train_image_generator, train_mask_generator)
    #To generate a generator instead of a zip object and handle the error
    train_generator = image_gen(all_df_train,batch_size = BATCH_SIZE)
    val_generator = image_gen(all_df_val,batch_size = BATCH_SIZE)
    
    
    #for msk_ in train_generator:
    #    print(msk_[0].shape)
     #   break
    #val_generator = zip(val_image_generator, val_mask_generator)
    #val_generator = new_train_gen(val_image_generator, val_mask_generator)

    
    ##Training
    NO_OF_TRAINING_IMAGES = len(os.listdir(train_image_brain2_jpg))
    NO_OF_VAL_IMAGES = len(os.listdir(validation_image_brain2_jpg))
    
    weights_path = 'D:\\unet\\weight_path'
        
    m = unet(pretrained_weights = r"D:/unet/unet_membrane.hdf5",input_size = (256,256,1))
    opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    m.compile(loss = tf.keras.losses.BinaryCrossentropy(),
              optimizer = opt,
              metrics= ['accuracy'])
    #checkpoint = ModelCheckpoint(weights_path, monitor='accuracy', 
    #                         verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('./log.out', append=True, separator=';')
    
    #earlystopping = EarlyStopping(monitor = 'accuracy', verbose = 1,
    #                          min_delta = 0.01, patience = 3, mode = 'max')
    root_logdir = os.path.join(os.curdir, "logs_brain_accuracy")
    os.makedirs(root_logdir,exist_ok=True)
    run_logdir = get_run_logdir()
    
    #from tensorflow.keras.callbacks import LambdaCallback    
    
    #callbacks_list = [csv_logger, earlystopping,TensorBoard(run_logdir)]
    #callbacks_list.set_model(m)
    results = m.fit(train_generator, epochs = NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data = val_generator, 
                          validation_steps = (NO_OF_VAL_IMAGES//BATCH_SIZE)
                             )
                          #callbacks = callbacks_list)
    m.save('Model_unet.h5')
    
    