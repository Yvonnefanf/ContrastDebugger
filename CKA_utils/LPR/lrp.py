import numpy as np
import pickle

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


def get_model_params(model):
	names, activations, weights = ['input_1'], [model.input, ], [0,]

	for layer in model.layers:
		name = layer.name if layer.name != 'predictions' else 'fc_out'
		names.append(name)
		activations.append(layer.output)
		weights.append(layer.get_weights())

	return names, activations, weights


class LayerwiseRelevancePropagation:

	def __init__(self, X, Y, model):
		self.model = model
		self.X = X
		self.Y = Y
		self.alpha = 1
		self.beta = self.alpha -1
		self.epsilon = 1e-9
		self.names, self.activations, self.weights = get_model_params(self.model)
		self.num_layers = len(self.names)
		self.relevance= self.compute_relevances()
		self.lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])
		#self.get_output = K.function(inputs=[self.model.input, ], outputs=[self.model.output, ])
		#self.get_presoft_output = K.function(inputs=[self.model.input, ], outputs=[self.model.layers[self.num_layers-3].output, ])

	def compute_relevances(self):
		#############Compute LRP from  assigned index of the output ########################
		#r= self.model.layers[self.num_layers-3].output  #pre-softmax
		r = self.model.output	
		#index = K.argmax(r,axis=1)
		#ref = K.variable(tf.zeros([16], tf.float32), name='ref')
		#x = tf.scatter_update(ref,index,1)
		#x = tf.reshape(x, [1,16])
		#r = tf.math.multiply(r,x)
		####################################################################################

		# Trace model
		for i in range(self.num_layers-2, -1, -1):

			if 'dense' in self.names[i + 1]:
				#print('===================== compute_relevances fc=====================')
				r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
				#print('fc Relevances: ', r)

			elif 'flatten' in self.names[i + 1]:
				#print('=====================compute_relevances flatten=====================')
				r = self.backprop_flatten(self.activations[i], r)
				#print('########flatten Relevances: ######## ', r)
			elif 'conv' in self.names[i + 1]:
				#print('=====================compute_relevances conv=====================')
				r = self.backprop_conv(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
				#print('########conv Relevances: ######## ', r)
		return r

	def backprop_fc(self, w, b, a, r):
		'''
		w: weights
		b: bias 
		a: activation output
		r: relevance score
		'''

		w_p = K.maximum(w, 0.)
		#w_p= tf.convert_to_tensor(w)
		z_p = K.dot(a, w_p)+ self.epsilon	
		s_p = r / z_p
		c_p = K.dot(s_p, K.transpose(w_p))

		w_n = K.minimum(w, 0.)
		b_n = K.maximum(b, 0.)
		z_n = K.dot(a, w_n) - self.epsilon
		s_n = r / z_n
		c_n = K.dot(s_n, K.transpose(w_n))
		#print('===================== backprop_fc finished ==========================')
		
		return a * (self.alpha * c_p - self.beta * c_n)

	def backprop_flatten(self, a, r):
		'''
		a: activation output
		r: relevance score
		'''		
		shape = a.get_shape().as_list()
		shape[0] = -1
		#print('===================== backprop_flatten finished ==========================')

		return K.reshape(r, shape)

	def backprop_conv(self, w, b, a, r):

		w_p = K.maximum(w, 0)
		#w_p = tf.convert_to_tensor(w)

		z_p = K.conv1d(a, kernel=w_p, strides=1, padding='valid') + self.epsilon 
		s_p = r / z_p
		c_p = keras.backend.tf.contrib.nn.conv1d_transpose(value= s_p, filter= w_p, output_shape= K.shape(a), stride= 1, padding='VALID')
		w_n = K.minimum(w, 0.)
		z_n = K.conv1d(a, kernel=w_n, strides=1, padding='valid') - self.epsilon
		s_n = r / z_n
		c_n = keras.backend.tf.contrib.nn.conv1d_transpose(value= s_n,filter= w_n,output_shape= K.shape(a),stride= 1,padding='VALID')
		#print('===================== backprop_conv finished ==========================')
		
		return a * (self.alpha * c_p - self.beta * c_n)
	
	def run_lrp(self):
		#lrp_runner = K.function(inputs=[self.model.input, ], outputs=[self.relevance, ])
		relevance = self.lrp_runner([self.X, ])[0]
		relevance = relevance.reshape(self.X.shape[0],-1)
		return relevance

if __name__ == '__main__':

    X = np.array(pickle.load(open('CSI/exp1/Test_X.pickle', 'rb')))
    Y = np.array(pickle.load(open('CSI/exp1/Test_y.pickle', 'rb')))

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = load_model('models/32CNN.h5')
    model.summary()
    relevance_score = LayerwiseRelevancePropagation(X, Y, model).run_lrp()


    print(relevance_score[0])
    #Save relevance score as pickle
    #pickle_out = open('Relevances.pickle', 'wb')
    #pickle.dump(relevance_score, pickle_out)
    #pickle_out.close()