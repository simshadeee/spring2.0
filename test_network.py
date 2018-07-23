# USAGE
# python test_network.py --model santa_not_santa.model --image images/examples/santa_01.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils import paths

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
print("[INFO] loading network...")
model = load_model(args["model"])

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (28, 28))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "normal" else 0
	labels.append(label)

predictions = []
for i in range(len(data)):
	image = data[i]
	orig = image.copy()
	image = cv2.resize(image, (28, 28))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	(notSanta, santa) = model.predict(image)[0]
	predictions.append(1 if santa > notSanta else 0)

def get_stats(labels, predictions):
	precision_count = 0
	normal_count = labels.count(1)
	broken_count = labels.count(0)
	recall_normal_count = 0
	recall_broken_count = 0

	for i in range(len(labels)):
		precision_count += labels[i] == predictions[i]
		if labels[i] and predictions[i]: recall_normal_count += 1
		if not (labels[i] or predictions[i]): recall_broken_count += 1

	precision = float(precision_count) / len(labels) if len(labels) else 0.0
	recall_normal = float(recall_normal_count) / normal_count if normal_count else 0.0
	recall_broken = float(recall_broken_count) / broken_count if broken_count else 0.0

	return precision, recall_normal, recall_broken

precision, recall_normal, recall_broken = get_stats(labels, predictions)

print('Precision: ' + str(precision))
print('Recall (Normal): ' + str(recall_normal))
print('Recall (Broken): ' + str(recall_broken))


	# if santa > notSanta:
	# 	print(imagePath)
	# 	print(santa)

# for imagePath in imagePaths:
# 	# load the image
# 	print(imagePath)
# 	image = cv2.imread(imagePath)
# 	orig = image.copy()
#
# 	# pre-process the image for classification
# 	image = cv2.resize(image, (28, 28))
# 	image = image.astype("float") / 255.0
# 	image = img_to_array(image)
# 	image = np.expand_dims(image, axis=0)
#
# 	# load the trained convolutional neural network
#
#
# 	# classify the input image
# 	(notSanta, santa) = model.predict(image)[0]
#
# 	# build the label
#
# 	if santa > notSanta: #classified not broken
# 		#check if the path is in the santa folder
# 		print(imagePath)
# 		print(santa)


	#label = "Working" if santa > notSanta else "Broken"
	#labels.append(label)
#print("working",labels.count("Working"))
#print("Not Working",labels.count("Broken"))
