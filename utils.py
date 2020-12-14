"""
Set of useful functions used across extraction and testing files
"""
## Written by Ryan Smith
## University College Dublin
## ryan.smith@ucdconnect.ie

import os
import sys
import glob
import h5py
import random
import numpy as np
import Augmentor

import torch
import torchvision.transforms as T
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

# other imports in respective functions as they are only called once for efficiency


def load_pretrained(model_name, include_top=False, weights="imagenet"):
    """
    Loads the pre-trained CNN model (mobilenet or xception)

    Args:
        model_name = (string) name of the CNN
        include_top = (bool) include the fully connected layer or not
        weights = (string) whether to use the pretrained weights from imagenet or
                           train from randomised weights

    Returns:
        model = Pre-trained CNN model
    """
    # pre-trained CNNs
    if model_name == "mobilenet":
        from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
        base_model = MobileNetV2(include_top=include_top, weights=weights, 
                                 input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
        try:
            model = Model(base_model.input, base_model.get_layer('custom').output)	  
        except:
            model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
        image_shape = (224, 224)
    elif model_name == "xception":
        from tensorflow.python.keras.applications.xception import Xception
        base_model = Xception(weights=weights)
        try:
            model = Model(base_model.input, base_model.get_layer('avg_pool').output)	  
        except:
            model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
        image_shape = (299, 299)
    else:
        model = None

    if model is None:
        raise Exception("Error: Model not loaded. \nExiting...")
        sys.exit()

    print("base model and model loaded...\n")
    return model, image_shape

def save_list_h5(path, list_to_save):
    """
    Save the passes list to the path as a H5 datafile

    Args:
        path = (string) where to save the H5 file (include .h5 extension)
        list_to_save = (list) the vector list to save in the file
    """
    with open(h5py.File(path), 'w') as hfile:
        hfile.create_dataset('dataset_1', data=np.array(list_to_save))

def single_input_extraction(model_name, train_path, train_labels, imaug=False):
    """
    Extract the image features given a folder with image samples using a single
        image as the inputs

    Args:
        train_path = (string) path to the training folder
        train_labels = (list) names of the class labels
        image_size = (tuple) size to resize the images for preprocessing
        imaug = (bool) to perform image augmentation or not, augmented images are 
                stored in a subfolder "output"

    Returns:
        features = (list) vector of image feature vectors
        labels = (list) vector of image labels
    """
    from tensorflow.python.keras.preprocessing import image
    if model_name == "mobilenet":
        from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "xception":
        from tensorflow.python.keras.applications.xception import preprocess_input

    model, image_shape = load_pretrained(model_name)

    features = []
    labels = []

    #if imaug == "process":
    #    # Image Augmentation
    #    for k in range(len(train_labels)):
    #        p = Augmentor.Pipeline(train_path+os.sep+train_labels[k])
    #        p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    #        p.zoom(probability=0.5, min_factor=1.05, max_factor=1.5)
    #        p.crop_random(probability=0.5, percentage_area=0.85)
    #        p.random_brightness(probability=0.5, max_factor=1.2, min_factor=0.8)
    #        #p.flip_left_right(probability=0.5)
    #        #p.flip_top_bottom(probability=0.5)
    #        p.sample(2000)
    #        del p

    # loop over all the labels and images in the folder
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        paths = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
        if imaug:
            paths += glob.glob(cur_path + "/oversample/*.png") + \
                    glob.glob(cur_path + "/oversample/*.jpg")
        count = 1
        for image_path in paths:
            img = image.load_img(image_path, target_size=image_shape)
            x = transform = transform(imaug, image_shape, img)
            x = preprocess_input(x)
            feature = model.predict(x)
            # Ensure features are in vector form
            flat = feature.flatten()
            features.append(flat)
            labels.append(label)
            print("processed - " + str(count))
            count += 1
        print("completed label - " + label)
    return features, labels

def two_input_extraction(model_name, train_path, train_labels, imaug=False):
    """
    Extract the image features given a folder with image samples using two
        images as the inputs

    Args:
        train_path = (string) path to the training folder
        train_labels = (list) names of the class labels
        image_size = (tuple) size to resize the images for preprocessing
        imaug = (bool) to perform image augmentation or not, augmented images are 
                stored in a subfolder "output"

    Returns:
        features = (list) vector of image feature vectors
        labels = (list) vector of image labels
    """
    from tensorflow.python.keras.preprocessing import image
    if model_name == "mobilenet":
        from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "xception":
        from tensorflow.python.keras.applications.xception import preprocess_input

    model, image_shape = load_pretrained(model_name)

    features = []
    labels = []

    # loop over all the labels and images in the folder
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        count = 1
        file_names = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
        if imaug:
            file_names += glob.glob(cur_path + "/oversample/*.jpg") +\
                          glob.glob(cur_path + "/oversample/*.png")
        file_names = sorted(file_names)
        for image_path1, image_path2 in zip(file_names, file_names[1:]):
            diff = int(image_path2[-16:-4]) - int(image_path1[-16:-4])
            if  diff < 20 and diff > 0:
                #try:
                img1 = image.load_img(image_path1, target_size=image_shape)
                img2 = image.load_img(image_path2, target_size=image_shape)

                imgs = transform(imaug, image_shape, img1, img2)

                x = imgs[0]
                y = imgs[1]

                x = preprocess_input(x)
                y = preprocess_input(y)
                # Tensorflow models are channels last
                x = np.transpose(x, (2,1,0))[None]
                y = np.transpose(y, (2,1,0))[None]

                featx = model.predict(x)
                featy = model.predict(y)

                features.append([featx.flatten(), featy.flatten()])
                labels.append(label)
                print ("processed - " + str(count))
                count += 1
                #except:
                #    pass

    return features, labels

def three_input_extraction(model_name, train_path, train_labels, imaug=False):
    """
    Extract the image features given a folder with image samples using three
        images as the inputs

    Args:
        train_path = (string) path to the training folder
        train_labels = (list) names of the class labels
        image_size = (tuple) size to resize the images for preprocessing
        imaug = (bool) to perform image augmentation or not, augmented images are 
                stored in a subfolder "output"

    Returns:
        features = (list) vector of image feature vectors
        labels = (list) vector of image labels
    """
    from tensorflow.python.keras.preprocessing import image
    if model_name == "mobilenet":
        from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "xception":
        from tensorflow.python.keras.applications.xception import preprocess_input

    model, image_shape = load_pretrained(model_name)

    features = []
    labels = []

    # loop over all the labels and images in the folder
    for i, label in enumerate(train_labels):
        cur_path = train_path + "/" + label
        count = 1
        file_names = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
        if imaug:
            file_names += glob.glob(cur_path + "/oversample/*.jpg") +\
                          glob.glob(cur_path + "/oversample/*.png")
        file_names = sorted(file_names)
        for image_path1, image_path2 in zip(file_names, file_names[1:]):
            if int(image_path2[-16:-4]) - int(image_path1[-16:-4]) < 20:
                try:
                    img1 = image.load_img(image_path1, target_size=image_shape)
                    img2 = image.load_img(image_path2, target_size=image_shape)
                    img3 = image.load_img('data/flow/train/'+label+'/'+image_path1[-16:-4]+'.png',
                                          target_size=image_shape)

                    imgs = transform(imaug, image_shape, img1, img2, img3)

                    x = imgs[0]
                    y = imgs[1]
                    z = imgs[2]

                    x = preprocess_input(x)
                    y = preprocess_input(y)
                    z = preprocess_input(z)
                    # Tensorflow models are channels last
                    x = np.transpose(x, (2,1,0))[None]
                    y = np.transpose(y, (2,1,0))[None]
                    z = np.transpose(z, (2,1,0))[None]

                    featx = model.predict(x)
                    featy = model.predict(y)
                    featz = model.predict(z)

                    features.append([featx.flatten(), featy.flatten(), featz.flatten()])
                    labels.append(label)
                    print ("processed - " + str(count))
                    count += 1
                except:
                    pass
    return features, labels

def transform(augment, image_shape, *args):
    if augment:
        preprocess_transform = T.Compose([
            #T.ToPILImage(),
            T.RandomRotation(degrees=15),
            T.RandomResizedCrop(size=image_shape, scale=(0.8, 1.2)),
            #T.ColorJitter(0.3, 0.2, 0.2, 0.2),
            T.ToTensor()
        ])
    else:
        preprocess_transform = T.Compose([
            #T.ToPILImage(),
            T.Resize(size=image_shape),
            T.ToTensor()
        ])

    imgs = []
    for img in args:
        seed = random.randint(0, 2**32)
        random.seed(seed)
        imgs.append(preprocess_transform(img).numpy())
    return imgs
