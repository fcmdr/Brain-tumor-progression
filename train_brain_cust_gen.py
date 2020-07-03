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

def create_folders(base_folder):
    """Create a "base_folder" (the physical path in which you want to create the subfolders) then this base folder will be used to create the train,validation and test folders for the images and the corresponding mask"""
    global train_image_brain
    train_image_brain = os.path.join(base_folder,'train_images')
    os.mkdir(train_image_brain)
    global validation_image_brain
    validation_image_brain = os.path.join(base_folder,'val_images')
    os.mkdir(validation_image_brain)
    global test_image_brain
    test_image_brain = os.path.join(base_folder,'test_images')
    os.mkdir(test_image_brain)
    
    global train_image_brain2
    train_image_brain2 = os.path.join(base_folder,'train_images2')
    os.mkdir(train_image_brain2)
    global validation_image_brain2
    validation_image_brain2 = os.path.join(base_folder,'val_images2')
    os.mkdir(validation_image_brain2)
    global test_image_brain2
    test_image_brain2 = os.path.join(base_folder,'test_images2')
    os.mkdir(test_image_brain2)
    
    #Create masks folders
    global train_masks_brain
    train_masks_brain = os.path.join(base_folder,'train_masks')
    os.mkdir(train_masks_brain)
    global validation_masks_brain
    validation_masks_brain = os.path.join(base_folder,'val_masks')
    os.mkdir(validation_masks_brain)
    global test_masks_brain
    test_masks_brain = os.path.join(base_folder,'test_masks')
    os.mkdir(test_masks_brain)

    #Create masks folders with images instead of dicom files
    global train_masks_brain2
    train_masks_brain2 = os.path.join(base_folder,'train_masks2')
    os.mkdir(train_masks_brain2)
    global validation_masks_brain2
    validation_masks_brain2 = os.path.join(base_folder,'val_masks2')
    os.mkdir(validation_masks_brain2)
    global test_masks_brain2
    test_masks_brain2 = os.path.join(base_folder,'test_masks2')
    os.mkdir(test_masks_brain2)
    
class Dataset():
    def __init__(self,base_folder):
        self.train_image_brain = os.path.join(base_folder,'train_images')
        self.validation_image_brain = os.path.join(base_folder,'val_images')
        self.test_image_brain = os.path.join(base_folder,'test_images')
        
        self.train_image_brain2 = os.path.join(base_folder,'train_images2')
        self.validation_image_brain2 = os.path.join(base_folder,'val_images2')
        self.test_image_brain2 = os.path.join(base_folder,'test_images2')
        
        self.train_masks_brain = os.path.join(base_folder,'train_masks')
        self.validation_masks_brain = os.path.join(base_folder,'val_masks')
        self.test_masks_brain = os.path.join(base_folder,'test_masks')
        
        self.train_masks_brain2 = os.path.join(base_folder,'train_masks2')
        self.validation_masks_brain2 = os.path.join(base_folder,'val_masks2')
        self.test_masks_brain2 = os.path.join(base_folder,'test_masks2')
    def create(self):
        folders = [self.train_image_brain,\
                   self.validation_image_brain,\
                   self.test_image_brain,\
                   
                   self.train_image_brain2,\
                   self.validation_image_brain2,\
                   self.test_image_brain2,\
                   
                   self.train_masks_brain,\
                   self.validation_masks_brain,\
                   self.test_masks_brain,\
                   
                   self.train_masks_brain2,\
                   self.validation_masks_brain2,\
                   self.test_masks_brain2,\
                  ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    #check __init__py
    #Ideally this function should be a method of the dataset class (which is not the case here)
    #Relative path !!
    
def populate_folders_with_images(my_dataset):
    #Here no need to use global variable thanks to class
    train_image_folder = my_dataset.train_image_brain
    train_masks_folder = my_dataset.train_masks_brain
    
    for image in all_df_train['dcm_path']:
        shutil.copy(image, train_image_folder)
    for mask in all_df_train['Mask']:
        shutil.copy(mask, train_masks_folder)

    #Move validation
    for image_val in all_df_val['dcm_path']:
        shutil.copy(image_val, validation_image_folder)
    for mask_val in all_df_val['Mask']:
        shutil.copy(mask_val, validation_masks_folder)

    #Move test
    for image_test in all_df_test['dcm_path']:
        shutil.copy(image_test, test_image_folder)
    for mask_test in all_df_test['Mask']:
        shutil.copy(mask_test, test_mask_folder)

        
def move_dicom_other_folder():
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
        os.mkdir(jpg_folder_path)
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
    
    #Create the folder named "folder_brain" inside the base_dir_brain folder whose path is : 'D:\\brain_small'    
    base_dir_brain = 'D:\\brain_small'               
    folder_brain = os.path.join(base_dir_brain, 'folder_brain') 
    
    all_df = pd.DataFrame({'dcm_path': sorted(list(glob2.glob(os.path.join(folder_brain,"*T2reg*.dcm"))))})#In the 'all_df' dataframe create a variable 'dcm_path' built as a list of the "T2reg" dcm_files  and sort the list by patients and slices
    all_df['slice_id']=all_df['dcm_path'].map(lambda x: (x.split('_')[-1]))#Extract "slice_id" from the "dcm_path variable"
    all_df['Type'] = np.where(all_df['dcm_path'].str.contains('Mask'), #create a variable type 'Mask' for the files masks, "Image" for the images
    'Mask', "Image")
    #Create Mask variable containing masks paths of associated images
    all_df['Mask'] = sorted(list(glob2.glob(os.path.join(folder_brain,"*Mask*.dcm"))))
    
    #Create a sub dataframe to test stuffs on the model easily
    #shuffle the dataframe
    all_df_sub = all_df.sample(frac=1)[:30]
    
    all_df_train,all_df_val,all_df_test = split_dataframe(all_df_sub,0.7,0.9)
    size_all_df = len(all_df_sub)
    size_all_df_train = len(all_df_train)
    size_all_df_val = len(all_df_test)
    
    #Create folders from base folder
    base_folder_brain = 'D:\\fit_gen_brain_test'
    create_folders(base_folder_brain)
    
    #Copy dicom files in another folder
    #Copy train images into the train_image_brain2 (images instead of dicom files)
    
    fouad_dataset = Dataset(base_folder_brain)
    
    populate_folders_with_images(fouad_dataset)
    
    move_dicom_other_folder()

    
    #Convert dcm files into images
    folder_list_to_transform =[train_image_brain2,validation_image_brain2,test_image_brain2,train_masks_brain2,validation_masks_brain2,test_masks_brain2]
    #Transform dicom files into jpg images 
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
      
    NO_OF_EPOCHS = 1
    BATCH_SIZE = 1
    
    train_image_brain2_jpg = "D:\\fit_gen_brain_sub\\train_images2_jpg"
    validation_image_brain2_jpg = "D:\\fit_gen_brain_sub\\val_images2_jpg"
    test_image_brain2_jpg = "D:\\fit_gen_brain_sub\\test_images2_jpg"
        
    train_masks_brain2_jpg = "D:\\fit_gen_brain_sub\\train_masks2_jpg"
    validation_masks_brain2_jpg = "D:\\fit_gen_brain_sub\\val_masks2_jpg"
    test_masks_brain2_jpg = "D:\\fit_gen_brain_sub\\test_masks2_jpg"
        
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_image_generator = train_datagen.flow_from_directory(train_image_brain2_jpg,batch_size = BATCH_SIZE)
    train_mask_generator = train_datagen.flow_from_directory(train_masks_brain2_jpg,batch_size = BATCH_SIZE)
    val_image_generator = val_datagen.flow_from_directory(validation_image_brain2_jpg, batch_size = BATCH_SIZE)
    val_mask_generator = val_datagen.flow_from_directory(validation_masks_brain2_jpg,batch_size = BATCH_SIZE)
    
    #Build a couple of image and its corresponding mask
    #train_generator = zip(train_image_generator, train_mask_generator)
    #To generate a generator instead of a zip object and handle the error
    #######modify all_df_train to have the dataframe with the corresponding images !!!!!##########
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
     #                        verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('./log.out', append=True, separator=';')
    
    earlystopping = EarlyStopping(monitor = 'accuracy', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')
    root_logdir = os.path.join(os.curdir, "logs_brain_accuracy")
    os.makedirs(root_logdir,exist_ok=True)
    run_logdir = get_run_logdir()
    
    from tensorflow.keras.callbacks import LambdaCallback    
    
    callbacks_list = [csv_logger, earlystopping,TensorBoard(run_logdir)]
    #callbacks_list.set_model(m)
    results = m.fit(train_generator, epochs = NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data = val_generator, 
                          validation_steps = (NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks = callbacks_list)
    m.save('Model_unet.h5')
    
    