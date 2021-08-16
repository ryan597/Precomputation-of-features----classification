"""
Set of useful functions used across extraction and testing files
"""
## Written by Ryan Smith
## University College Dublin
## ryan.smith@ucdconnect.ie

import sys
import glob
import random
import h5py
import numpy as np

# Suppress tensorflow depreciation warnings
import warnings
warnings.filterwarnings('ignore')

import torchvision.transforms as T
import albumentations as A
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

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
    Save the passed list to the path as a H5 datafile

    Args:
        path = (string) where to save the H5 file (include .h5 extension)
        list_to_save = (list) the vector list to save in the file
    """
    hfile = h5py.File(path, 'w')
    hfile.create_dataset('dataset_1', data=np.array(list_to_save))
    hfile.close()

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

    # loop over all the labels and images in the folder
    for label in train_labels:
        cur_path = train_path + "/" + label
        paths = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")

        count = 1
        for image_path in paths:
            img = image.load_img(image_path, target_size=image_shape)
            img = transform(False, image_shape, np.array(img))
            # Expects a 4D float array
            x = img['image'][None].astype('float')
            x = preprocess_input(x)
            feature = model.predict(x)
            # Ensure features are in vector form
            flat = feature.flatten()
            features.append(flat)
            labels.append(label)
            print("processed - " + str(count))
            count += 1

        if imaug:
            oversample = 0
            while oversample < 2000:
                for image_path in paths:
                    if oversample < 2000 and np.random.random(1) < 0.3:
                        img = image.load_img(image_path, target_size=image_shape)
                        img = transform(True, image_shape, np.array(img))
                        # Expects a 4D float array
                        x = img['image'][None].astype('float')
                        x = preprocess_input(x)
                        feature = model.predict(x)
                        # Ensure features are in vector form
                        flat = feature.flatten()
                        features.append(flat)
                        labels.append(label)
                        print("processed - " + str(count))
                        count += 1
                        oversample += 1
                    elif oversample >= 2000:
                        break
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
    for label in train_labels:
        cur_path = train_path + "/" + label
        count = 1
        file_names = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
        file_names = sorted(file_names)

        for image_path1, image_path2 in zip(file_names, file_names[1:]):
            # get the image names and compare
            diff = int(image_path2[-16:-4]) - int(image_path1[-16:-4])
            if  diff < 20 and diff > 0:
                img1 = image.load_img(image_path1, target_size=image_shape)
                img2 = image.load_img(image_path2, target_size=image_shape)

                imgs = transform(imaug, image_shape, np.array(img1), np.array(img2))

                x = imgs['image'][None].astype('float')
                y = imgs['image2'][None].astype('float')

                x = preprocess_input(x)
                y = preprocess_input(y)

                featx = model.predict(x)
                featy = model.predict(y)

                features.append(np.array([featx, featy]).flatten())
                labels.append(label)
                print ("processed - " + str(count))
                count += 1

        if imaug == True:
            oversample = 0
            while oversample < 2000:
                for image_path1, image_path2 in zip(file_names, file_names[1:]):
                    diff = int(image_path2[-16:-4]) - int(image_path1[-16:-4])
                    if  diff < 20 and diff > 0:
                        if oversample < 2000 and np.random.random(1) < 0.3: # random accept or reject this image pair
                            img1 = image.load_img(image_path1, target_size=image_shape)
                            img2 = image.load_img(image_path2, target_size=image_shape)

                            imgs = transform(True, image_shape, np.array(img1), np.array(img2))

                            x = imgs['image'][None].astype('float')
                            y = imgs['image2'][None].astype('float')

                            x = preprocess_input(x)
                            y = preprocess_input(y)

                            featx = model.predict(x)
                            featy = model.predict(y)

                            features.append(np.array([featx, featy]).flatten())
                            labels.append(label)
                            print ("processed - " + str(count))
                            count += 1
                            oversample +=1
                        elif oversample >= 2000:
                            break

        print("completed label - " + label)
    return features, labels

def two_input_IR_FLO(model_name, train_path, train_labels, imaug=False):
    from tensorflow.python.keras.preprocessing import image
    import os.path
    if model_name == "mobilenet":
        from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
    elif model_name == "xception":
        from tensorflow.python.keras.applications.xception import preprocess_input

    model, image_shape = load_pretrained(model_name)

    features = []
    labels = []

    # loop over all the labels and images in the folder
    for label in train_labels:
        cur_path = train_path + "/" + label
        path_OF = "data/flow/train/" + label
        count = 1
        file_names_IR = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
        file_names_IR = sorted(file_names_IR)
        file_names_OF = glob.glob(path_OF + "/*.jpg") + glob.glob(path_OF + "/*.png")
        file_names_OF = sorted(file_names_OF)

        for image_path_IR in file_names_IR:
            image_path_OF = image_path_IR[:5] + "flow" + image_path_IR[7:-4] + ".png"
            if os.path.exists(image_path_OF):
                img_IR = image.load_img(image_path_IR, target_size=image_shape)
                img_OF = image.load_img(image_path_OF, target_size=image_shape)

                imgs = transform(imaug, image_shape, np.array(img_IR), np.array(img_OF))

                x = imgs['image'][None].astype('float')
                y = imgs['image2'][None].astype('float')

                x = preprocess_input(x)
                y = preprocess_input(y)

                featx = model.predict(x)
                featy = model.predict(y)

                features.append(np.array([featx, featy]).flatten())
                labels.append(label)
                print ("processed - " + str(count))
                count += 1

        if imaug == True:
            oversample = 0
            while oversample < 2000:
                for image_path_IR in file_names_IR:
                    image_path_OF = image_path_IR[:5] + "flow" + image_path_IR[7:-4] + ".png"
                    if os.path.exists(image_path_OF) and oversample < 2000 and np.random.random(1) < 0.3:
                        img_IR = image.load_img(image_path_IR, target_size=image_shape)
                        img_OF = image.load_img(image_path_OF, target_size=image_shape)

                        imgs = transform(imaug, image_shape, np.array(img_IR), np.array(img_OF))

                        x = imgs['image'][None].astype('float')
                        y = imgs['image2'][None].astype('float')

                        x = preprocess_input(x)
                        y = preprocess_input(y)

                        featx = model.predict(x)
                        featy = model.predict(y)

                        features.append(np.array([featx, featy]).flatten())
                        labels.append(label)
                        print ("processed - " + str(count))
                        count += 1
                        oversample +=1
                    
                    elif oversample >= 2000:
                        break

        print("completed label - " + label)
    return features, labels

def transform(augment, image_shape, img1, img2=None):
    """
    Apply transformations to input images. Augmentations can also be applied.
    """
    additional_targets = {"image2":"image"} if type(img2) == np.ndarray else None

    if augment:
        preprocess_transform = A.Compose([
            A.Rotate(limit=20, p=0.6),
            A.RandomResizedCrop(image_shape[0], image_shape[1], scale=(0.8, 1.2)),
            #A.ColorJitter(0.3, 0.2, 0.2, 0.2),
            ],
            additional_targets = additional_targets
        )
    else:
        preprocess_transform = A.Compose([
            A.Resize(image_shape[0], image_shape[1]),
            ],
            additional_targets = additional_targets
        )
    transformed = preprocess_transform(image=img1, image2=img2)
    return transformed
