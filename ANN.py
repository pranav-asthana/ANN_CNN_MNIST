import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train)
print(y_train)

#Initializing Neural Network
classifier = Sequential()

# # Adding the input layer and the first hidden layer
# classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# # # Adding the second hidden layer
# # classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# # Adding the output layer
# classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# # Compiling Neural Network
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# # Fitting our model 
# classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)