## extract_CNN_features_imaug.py 
## A script to extract image features using a deep CNN
## Written by Daniel Buscombe,
## Northern Arizona University
## daniel.buscombe@nau.edu

# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

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
import numpy as np
import glob

import h5py
import os, sys, getopt
import json
import datetime
import time
import Augmentor

#==============================================================
if __name__ == '__main__':
	argv = sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv,"h:c:")
	except getopt.GetoptError:
		print('python extract_features_imaug.py -c conf_file')
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
	model_name    = config["model"]
	weights     = config["weights"]
	include_top   = config["include_top"]
	train_path    = config["train_path"]
	features_path   = config["features_path"]
	labels_path   = config["labels_path"]
	test_size     = config["test_size"]
	results     = config["results"]
	model_path    = config["model_path"]

	# start time
	print ("start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
	start = time.time()

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

	# path to training dataset
	train_labels = os.listdir(train_path)

	train_labels =[t for t in train_labels if not t.endswith('csv')]

	# encode the labels
	print ("encoding labels...")
	le = LabelEncoder()
	le.fit([tl for tl in train_labels])

	# Image Augmentation
	#for k in range(len(train_labels)):
	#	p = Augmentor.Pipeline(train_path+os.sep+train_labels[k])
	#	p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
	#	p.zoom(probability=0.3, min_factor=1.1, max_factor=1.5)
	#	p.crop_random(probability=0.25, percentage_area=0.8)
	#	#p.flip_left_right(probability=0.5)
	#	#p.flip_top_bottom(probability=0.5)
	#	p.sample(1000)
	#	del p

	# variables to hold features and labels
	features = []
	labels   = []

	# loop over all the labels in the folder
	count = 1
	for i, label in enumerate(train_labels):
		cur_path = train_path + "/" + label
		count = 1
		file_names = glob.glob(cur_path + "/*.jpg") + glob.glob(cur_path + "/*.png")
		file_names = sorted(file_names)
		for image_path1, image_path2 in zip(file_names, file_names[1:]):
			if int(image_path2[-16:-4]) - int(image_path1[-16:-4]) < 20:
				try:
					img1 = image.load_img(image_path1, target_size=image_size)
					img1 = image.img_to_array(img1)[:, :, 0][None]
					img2 = image.load_img(image_path2, target_size=image_size)
					img2 = image.img_to_array(img2)[:, :, 0][None]
					img3 = image.load_img('data/flow/train/'+label+'/'+image_path1[-16:-4]+'.png', target_size=image_size, color_mode='grayscale')
					img3 = image.img_to_array(img3)[:,:,0][None]
					x = np.concatenate((img1, img2, img3), 0)
					x = np.transpose(x, (2, 1, 0))[None]
					x = preprocess_input(x)
					feature = model.predict(x)
					flat = feature.flatten()
					features.append(flat)
					labels.append(label)
					print ("processed - " + str(count))
					count += 1
				except:
					pass

		print ("completed label - " + label)



	# encode the labels using LabelEncoder
	le = LabelEncoder()
	le_labels = le.fit_transform(labels)

	# get the shape of training labels
	print ("training labels: {}".format(le_labels))
	print ("training labels shape: {}".format(le_labels.shape))
	
	try:
		os.mkdir(os.getcwd()+os.sep+'out'+os.sep+model_name)
	except:
		pass
	
	# save features and labels
	h5f_data = h5py.File(features_path, 'w')
	h5f_data.create_dataset('dataset_1', data=np.array(features))

	h5f_label = h5py.File(labels_path, 'w')
	h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

	h5f_data.close()
	h5f_label.close()

	# save model and weights
	model_json = model.to_json()
	with open(model_path + str(test_size) + ".json", "w") as json_file:
	  json_file.write(model_json)

	# save weights
	model.save_weights(model_path + str(test_size) + ".h5")
	print("saved model and weights to disk..")

	print ("features and labels saved..")

	# end time
	end = time.time()
	print ("end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
