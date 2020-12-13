# keras imports
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.python.keras.applications.xception import Xception, preprocess_input
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.layers import Input


# other imports
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os, sys, getopt
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import glob


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
	model_name = config["model"]
	test_size     = config["test_size"]
	seed      = config["seed"]
	features_path   = config["features_path"]
	labels_path   = config["labels_path"]
	testfeatures_path   = config["test_features"]
	testlabels_path   = config["test_labels"]
	results     = config["results"]
	model_path = config["model_path"]
	train_path    = config["train_path"]
	test_path = config['test_path']
	num_classes   = config["num_classes"]
	classifier_path = config["classifier_path"]
	cm_path = config["cm_path"]
	include_top   = config["include_top"]
	weights     = config["weights"]


	# create the pretrained models
	if model_name == "vgg16":
		base_model = VGG16(weights=weights)
		try:
			model = Model(base_model.input, base_model.get_layer('fc1').output)	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
		image_size = (224, 224)
	elif model_name == "vgg19":
		base_model = VGG19(weights=weights)
		try:
			model = Model(base_model.input, base_model.get_layer('fc1').output)	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
		image_size = (224, 224)
	elif model_name == "resnet50":
		base_model = ResNet50(weights=weights)
		try:
		  model = Model(base_model.input, base_model.get_layer('avg_pool').output)	  	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)	  
		image_size = (224, 224)
	elif model_name == "inceptionv3":
		base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
		try:
			model = Model(base_model.input, base_model.get_layer('custom').output)	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
		image_size = (299, 299)
	elif model_name == "inceptionresnetv2":
		base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
		try:
			model = Model(base_model.input, base_model.get_layer('custom').output)	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
		image_size = (299, 299)
	elif model_name == "mobilenet":
		base_model = MobileNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
		try:
			model = Model(base_model.input, base_model.get_layer('custom').output)	  
		except:	  
			model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
		image_size = (224, 224)
	elif model_name == "xception":
		base_model = Xception(weights=weights)
		try:
			model = Model(base_model.input, base_model.get_layer('avg_pool').output)	  
		except:
			model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
		image_size = (299, 299)
	else:
		base_model = None

	print ("loaded base model and model...")


##############################################################################
	"""
	test_labels = os.listdir(test_path)
	
	test_labels =[t for t in test_labels if not t.endswith('csv')]
	# encode the labels
	print ("encoding labels...")
	le = LabelEncoder()
	le.fit([tl for tl in test_labels])
	# variables to hold features and labels
	features = []
	labels   = []

	# loop over all the labels in the folder
	count = 1
	for i, label in enumerate(test_labels):
		cur_path = test_path + "/" + label
		count = 1
		file_names = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
		file_names = sorted(file_names)
		for image_path1, image_path2 in zip(file_names, file_names[1:]):
			if int(image_path2[-16:-4]) - int(image_path1[-16:-4]) < 20:
				img1 = image.load_img(image_path1, target_size=image_size)
				img1 = image.img_to_array(img1)[None]
				img2 = image.load_img(image_path2, target_size=image_size)
				img2 = image.img_to_array(img2)[None]

				x = preprocess_input(img1)
				y = preprocess_input(img2)
				
				feature_1 = model.predict(x)
				feature_2 = model.predict(y)
				feature = np.concatenate((feature_1, feature_2), 1)

				flat = feature.flatten()
				features.append(flat)
				labels.append(label)
				print ("processed - " + str(count))
				count += 1
		print ("completed label - " + label)
	# encode the labels using LabelEncoder
	le = LabelEncoder()
	le_labels = le.fit_transform(labels)

	# save features and labels
	h5f_data = h5py.File(testfeatures_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(features))

	h5f_label = h5py.File(testlabels_path, 'w')
	h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

	h5f_data.close()
	h5f_label.close()
	"""

	# import features and labels
	h5f_data  = h5py.File(testfeatures_path, 'r')
	h5f_label = h5py.File(testlabels_path, 'r')

	features_string = h5f_data['dataset_1']
	labels_string   = h5f_label['dataset_1']

	features = np.array(features_string)
	labels   = np.array(labels_string)

	h5f_data.close()
	h5f_label.close()

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
	one_hot_labels = np.zeros((len(labels), 3))
	pred_prob = logmodel.predict_proba(features)
	for i, value in enumerate(labels):
		one_hot_labels[i, value] = 1
	bs = np.mean(np.sum((pred_prob - one_hot_labels)**2, axis=1))
	print(f"Brier Score: {bs}")


	f = open(results, "w")
	f.write("Accuracy:  {}\n".format(rank_1))
	f.write("Brier score:  {}\n".format(bs))
	f.write("{}\n".format(classification_report(labels,  preds)))
	f.close()


	from sklearn.metrics import roc_auc_score
	auc_ovr = roc_auc_score(labels, logmodel.predict_proba(features), multi_class='ovr')
	print("AUC ovr\t", auc_ovr)
	auc_ovo = roc_auc_score(labels, logmodel.predict_proba(features), multi_class='ovo')
	print("AUC ovo\t", auc_ovo)

	# display the confusion matrix
	print ("confusion matrix")

	# get the list of training lables
	classes = sorted(list(os.listdir(train_path)))
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
	plt.savefig(f"cm_{configfile}", bbox_inches='tight')
