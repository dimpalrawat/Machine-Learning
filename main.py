
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import random
	


def init_weights(shape):
	weights = tf.random_normal(shape , stddev=0.1)
	return tf.Variable(weights)

def shuffle_batch(arr):
	number = random.randint(0 , (arr.shape[0] - 511))
	return arr[number:number+512, :]

def init_bias(shape):
	bias = tf.constant(shape , dtype=tf.float32)
	return tf.Variable(bias)

def convert_to_hot_vectors(arr):
	num_labels = 10
	one_hot = np.eye(num_labels)[arr]
	return one_hot


if __name__ == '__main__':

	readdata = np.array(pd.read_csv("train.csv"))
	print (np.shape(readdata))


	#features and labels 

	traindata = readdata[0:41000 , :]

	testFeatures = readdata[41001:, 1:]
	testLabels = convert_to_hot_vectors(readdata[41001: , 0])

	num_of_feat  = 784
	num_of_labels = 10





	inputLayer = tf.placeholder(tf.float32 , [None , num_of_feat] )

	w1 = init_weights([num_of_feat , 200])

	b1 = init_bias([200])

	a1 = tf.nn.tanh(tf.matmul(inputLayer , w1) + b1)

	w2 = init_weights([200 , num_of_labels])

	b2 = init_bias([num_of_labels])

	outputLayer = tf.nn.softmax(tf.matmul(a1 , w2) + b2)


	y_ = tf.placeholder(tf.float32 , [None , 10])





	cross_entropy = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputLayer , y_))) 

	trainer = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

	predict_opt = tf.argmax(outputLayer , 1)

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)


	for i in range(10000):

		traindataBatch = shuffle_batch(traindata)

		trainFeatures = traindataBatch[:, 1:]
		trainLabels = convert_to_hot_vectors(traindataBatch[: , 0])

		sess.run(trainer , feed_dict={
			inputLayer : trainFeatures  , y_ : trainLabels 
			})

		accuracy = np.mean(np.argmax(testLabels , 1 ) == sess.run(predict_opt , feed_dict={

			inputLayer : testFeatures , y_ : testLabels
			}))

		print ("Epoch: %d Accuracy= %.4f% %" % (i, accuracy * 100)) 

































	#iterations




	
