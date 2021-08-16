## Written by Ryan Smith
## University College Dublin
## ryan.smith@ucdconnect.ie

import os
import json
import argparse
import pickle
import numpy as np
import h5py
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import utils

#==============================================================

if __name__ == '__main__':
    # Get command line argument for the config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Name of the config file inside ./conf/")
    parser.add_argument("-e", "--extract", default="y", help="Run extraction of test images, y/n. Default is y.")
    args = parser.parse_args()

    with open(os.getcwd()+os.sep+'conf'+os.sep+args.config+'.json') as f:
        config = json.load(f)

    # config variables
    model_name = config["model"]
    weights = config["weights"]
    include_top = config["include_top"]
    test_path = config["test_path"]
    test_features_path = config["test_features"]
    test_labels_path = config["test_labels"]
    results = config["results"]
    classifier_path = config["classifier_path"]
    extraction_func = config["extraction_func"]

#==============================================================
    if args.extract != "n":
        # encoding labels from train folder
        test_labels = os.listdir(test_path)
        le = LabelEncoder()
        le.fit(test_labels)

        # call the extraction function from the utils file using the string from conf
        features, labels = getattr(utils, extraction_func)(model_name,
                                                        test_path,
                                                        test_labels,
                                                        imaug=False)

        # encode the labels using LabelEncoder
        le = LabelEncoder()
        le_labels = le.fit_transform(labels)

        try:
            os.mkdir(os.getcwd()+os.sep+'out'+os.sep+model_name)
        except:
            pass

        # save features and labels as h5 files
        utils.save_list_h5(test_features_path, features)
        utils.save_list_h5(test_labels_path, le_labels)

        print("Extraction finished...\n")

#==============================================================

    # import features and labels
    h5f_data  = h5py.File(test_features_path, 'r')
    h5f_label = h5py.File(test_labels_path, 'r')

    features_string = h5f_data['dataset_1']
    labels_string   = h5f_label['dataset_1']

    features = np.array(features_string)
    labels   = np.array(labels_string)

    h5f_data.close()
    h5f_label.close()

    print(f"Features shape:\t{features.shape}")
    print(f"Labels shape:\t{labels.shape}")

    with open(classifier_path, 'rb') as file:
        logmodel = pickle.load(file)

    # Now test on the features
    rank_1 = 0
    for (feat, lab) in zip(features, labels):
        predictions = logmodel.predict_proba(np.atleast_2d(feat))[0]
        predictions = np.argsort(predictions)[::-1]

        # rank-1 prediction increment
        if lab == predictions[0]:
            rank_1 += 1
        #else:
        #	print("missclassified \t {}")
        #	print("True : {lab}\t Predicted : {predictions[0]}")
    rank_1 = (rank_1 / float(len(labels))) * 100
    print(f"\nrank_1 accuracy: {rank_1}")

    preds = logmodel.predict(features)
    print(classification_report(labels, preds))

    ##############################################################################

    # Brier score
    pred_prob = logmodel.predict_proba(features)
    one_hot_labels = np.zeros((len(labels), 3))
    for i, value in enumerate(labels):
        one_hot_labels[i, value] = 1
    bs = np.mean(np.sum((pred_prob - one_hot_labels)**2, axis=1))
    print(f"Brier Score: {bs}")


    f = open(results, "w")
    f.write("Accuracy:  {}\n".format(rank_1))
    f.write("Brier score:  {}\n".format(bs))
    f.write("{}\n".format(classification_report(labels,  preds)))
    f.close()

    auc_ovr = roc_auc_score(labels, pred_prob, multi_class='ovr')
    print("AUC ovr\t", auc_ovr)
    auc_ovo = roc_auc_score(labels, pred_prob, multi_class='ovo')
    print("AUC ovo\t", auc_ovo)


    # display the confusion matrix
    print ("confusion matrix")

    # get the list of test lables
    classes = sorted(list(os.listdir(test_path)))
    classes =[t for t in classes if not t.endswith('csv')]
    yclasses = ['true '+t for t in classes if not t.endswith('csv')]

    # plot the confusion matrix
    cm = confusion_matrix(labels, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.set(font_scale=2)
    sns.heatmap(cm,
                annot=True,
                cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True), cbar=False)

    tick_marks = np.arange(len(classes))+.5
    plt.xticks(tick_marks, classes, rotation=0,fontsize=20)
    plt.yticks(tick_marks, yclasses, rotation=0, fontsize=20)
    plt.savefig(f"figures/cm/cm_{args.config}", bbox_inches='tight')
