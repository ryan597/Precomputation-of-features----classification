## extract_CNN_features.py
## A script to extract image features using a deep CNN
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe@nau.edu

## Modified by Ryan Smith
## University College Dublin
## ryan.smith@ucdconnect.ie

import os
import json
import argparse

from sklearn.preprocessing import LabelEncoder

import utils

#==============================================================
if __name__ == '__main__':
    # Get command line argument for the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of the config file inside ./conf/")
    args = parser.parse_args()

    with open(os.getcwd()+os.sep+'conf'+os.sep+args.config+'.json') as f:
        config = json.load(f)

    # config variables
    model_name = config["model"]
    weights = config["weights"]
    include_top = config["include_top"]
    train_path = config["train_path"]
    features_path = config["features_path"]
    labels_path = config["labels_path"]
    results = config["results"]
    imaug = config["imaug"]
    extraction_func = config["extraction_func"]

    # encoding labels from train folder
    train_labels = os.listdir(train_path)
    le = LabelEncoder()
    le.fit(train_labels)

    # call the extraction function from the utils file using the string from conf
    features, labels = getattr(utils, extraction_func)(model_name,
                                                       train_path,
                                                       train_labels,
                                                       imaug=imaug)

    # encode the labels using LabelEncoder
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)

    try:
        os.mkdir(os.getcwd()+os.sep+'out'+os.sep+model_name)
    except:
        pass

    # save features and labels as h5 files
    utils.save_list_h5(features_path, features)
    utils.save_list_h5(labels_path, le_labels)

    print("Extraction finished...\n")
