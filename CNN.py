import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

# Keras backend is handled by Theano for this (why?)
K.set_image_dim_ordering('th')

# Fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Load data 
# TODO: using dataset (get into a numpy.ndarray)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape to be [samples][pixels][width][height] (!)
# The layers used for two-dimensional convolutions expect pixel values 
# with the dimensions [pixels][width][height]
# Pixels = 1 for MNIST because grayscale images
# 28*28 images size
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

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

	# First hidden layer. 32 feature maps of size 5*5. Activation function is ReLU.
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	# Pooling layer of size 2*2
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# # Second hidden layer. 15 feature maps of size 3*3. Activation function is ReLU.
	# model.add(Conv2D(15, (3, 3), activation='relu'))
	# # Pooling layer of size 2*2
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	# # Third hidden layer. 5 feature maps of size 2*2. Activation function is ReLU.
	# model.add(Conv2D(5, (2, 2), iactivation='relu'))
	# # Pooling layer of size 2*2
	# model.add(MaxPooling2D(pool_size=(2, 2)))

	# Regularization layer. Randomly excludes 20% of neurons in the layer to reduce overfitting
	model.add(Dropout(0.2))
	# Flatten to a 2D matrix
	model.add(Flatten())
	# A simple fully connected layer with 128 units.
	model.add(Dense(128, activation='relu'))

	# Add the output layer with the number of output units (classes)
	# Activation function is softmax (for probability-like classification)
	model.add(Dense(num_classes, activation='softmax'))

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
print("Accuracy over test set: %.2f%%" % scores[1]*100)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))