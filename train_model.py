## train_test_model.py 
## A script to train logistic reg. models for multi-class predictions from CNN extracted features
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe@nau.edu

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import h5py
import os, sys, getopt
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Random seed
SEED = 46

#==============================================================
if __name__ == '__main__':
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"h:c:")
    except getopt.GetoptError:
        print('python train_test_model.py -c conf_file')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('Example usage: python extract_features_imaug.py -c conf_mobilenet')
            sys.exit()
        elif opt in ("-c"):
            configfile = arg

    # load the user configs
    with open(os.getcwd()+os.sep+'conf'+os.sep+configfile+'.json') as f:    
      config = json.load(f)

    # config variables
    features_path   = config["features_path"]
    labels_path   = config["labels_path"]
    results     = config["results"]
    train_path    = config["train_path"]
    classifier_path = config["classifier_path"]

    # import features and labels
    h5f_data  = h5py.File(features_path, 'r')
    h5f_label = h5py.File(labels_path, 'r')

    features_string = h5f_data['dataset_1']
    labels_string   = h5f_label['dataset_1']

    features = np.array(features_string)
    labels   = np.array(labels_string)

    h5f_data.close()
    h5f_label.close()

    # verify the shape of features and labels
    print ("features shape: {}".format(features.shape))
    print ("labels shape: {}".format(labels.shape))

    print ("training started...")
    # split the training and testing data
    (trainData, trainLabels) = shuffle(features, labels, random_state=SEED)
    

    print ("splitted train and test data...")
    print ("train data  : {}".format(trainData.shape))
    print ("train labels: {}".format(trainLabels.shape))


    # use logistic regression as the model
    print ("creating model...")
    #model = LogisticRegression(C=0.5, dual=True, solver='liblinear', random_state=seed, class_weight='balanced', max_iter=100)
    model = LogisticRegression(C=0.5, random_state=SEED, class_weight='balanced', max_iter=1000)
    model.fit(trainData, trainLabels)

    # dump classifier to file
    print ("saving model...")
    pickle.dump(model, open(classifier_path, 'wb'))

    """
    # Can check model results on training data
    # use rank-1 and rank-5 predictions
    print ("evaluating model...")
    f = open(results, "w")

    # rank_1 -> true label is the most likely label
    rank_1 = 0
    # rank_5 -> true label is in top 5 likely labels
    rank_5 = 0

    # loop over test data
    for (label, features) in zip(trainLabels, trainData):
        # predict the probability of each class label and
        # take the top-5 class labels
        predictions = model.predict_proba(np.atleast_2d(features))[0]
        predictions = np.argsort(predictions)[::-1]

        # rank-1 prediction increment
        if label == predictions[0]:
            rank_1 += 1

        # rank-5 prediction increment
        if label in predictions:
            rank_5 += 1

    # convert accuracies to percentages
    rank_1 = (rank_1 / float(len(trainLabels))) * 100
    rank_5 = (rank_5 / float(len(trainLabels))) * 100

    # write the accuracies to file
    f.write("Rank-1: {:.2f}%\n".format(rank_1))
    f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

    # evaluate the model of test data
    preds = model.predict(trainData)

    # write the classification report to file
    f.write("{}\n".format(classification_report(trainLabels, preds)))
    f.close()

    # display the confusion matrix
    print ("confusion matrix")

    # get the list of training lables
    labels = sorted(list(os.listdir(train_path)))
    ##labels =[t for t in labels if not t.endswith('csv')]

    # plot the confusion matrix
    cm = confusion_matrix(trainLabels, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm,
                annot=True,
                cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True)) 

    tick_marks = np.arange(len(labels))+.5
    plt.xticks(tick_marks, labels, rotation=45,fontsize=5)
    plt.yticks(tick_marks, labels,rotation=45, fontsize=5)
    """
