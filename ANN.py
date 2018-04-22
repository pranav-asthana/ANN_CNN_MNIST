import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import math

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data 
# TODO: using dataset (get into a numpy.ndarray)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten 28*28 images to a 784 vector for each image (!)
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encode outputs - convert the labels to categorical 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Define baseline model
def baseline_model():
	# Create model - Sequential model is a linear stack of layers
	model = Sequential()

	# Add the first hidden layer (and 800 HU) and specify the input dimensions
	# Initialize the weight matrix using a normal distribution
	# Activation function h(a) is the ReLU function 
	model.add(Dense(800, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))

	# # 2nd hidden layer (100 HU)
	# model.add(Dense(100, kernel_initializer='normal', activation='relu'))

	# # 3rd hidden layer (20 HU)
	# model.add(Dense(20, kernel_initializer='normal', activation='relu'))

	# Add the output layer with the number of output units (classes)
	# Activation function is softmax (for probability-like classification)
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

	# Compile model - given the loss function and gradient descent optimizer type
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# Build the model
model = baseline_model()

# Fit the model
# Given data, labels, validation_data, number of iterations, verbose output lines
old_validation_error = math.inf
for i in range(4000):
	value = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)
	validation_error = value.history['val_loss'][0]
	if (validation_error - old_validation_error) > (0.01*old_validation_error):
		break
	old_validation_error = validation_error

# Final evaluation of the model
# Given test data, test data labels, verbose output lines
scores = model.evaluate(X_test, y_test, verbose=0)

# Calculate the accuracy and baseline error of the model
print("Accuracy over test set: %.2f%%" % (scores[1]*100))
print("Baseline Error: %.2f%%" % (100-scores[1]*100))