from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Merge
from keras.optimizers import SGD
from keras.utils.visualize_util import plot

import numpy as np
import matplotlib.pyplot as plt

# Create the neural network model
model = Sequential([
        Dense(2, input_dim=2),
        Activation('sigmoid'),
        Dense(2),
        Activation('softmax')])

# Compile the model with an optimiser
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

# Create training cases for an XOR function
x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[1,0], [0,1], [0,1], [1,0]])

# Create a map of points to plot the function stored in the neural network
width = 3
height = 3
X = [[x1, x2] for x2 in np.linspace(-(width-1)/2, 1+(width-1)/2, 201) for 
	x1 in np.linspace(-(height-1)/2, 1+(height-1)/2, 201)]

# Create a plot to visualise the progress of training
epochList = [20, 50, 100, 500, 1000, 5000, 10000]

# The figure is going to have its width equal to the sqrt of the number of axes
# we wish to plot. The other side will be the smallest number such that all 
# axes will fit. This is the width plus the ceiling of (remainder/width)
plotWidth = int(np.sqrt(len(epochList)))
remainder = len(epochList) - plotWidth**2

fig, axs = plt.subplots(plotWidth + (remainder-1)//plotWidth+1, plotWidth)
trained = 0

for i in range(0, len(epochList)):
	axs[int(i/plotWidth), i%plotWidth].set_xlim(-(width-1)/2, 1+(width-1)/2)
	axs[int(i/plotWidth), i%plotWidth].set_ylim(-(height-1)/2, 1+(height-1)/2)
	axs[int(i/plotWidth), i%plotWidth].set_title("{0:d} Epochs".
		format(epochList[i]))

	# Train the network
	model.fit(x, y, batch_size=4, nb_epoch=epochList[i] - trained, verbose=0)
	trained += epochList[i]

	# Run the trained neural network and colour points on a sliding scale
	# with blue for 0 and red for 1
	Y = model.predict(X)
	c = [(y[0], 0, y[1]) for y in Y]

	# Plot the output
	Xt = np.transpose(X)
	axs[int(i/plotWidth), i%plotWidth].scatter(Xt[0], Xt[1], color=c)
	# Add marks at (0,0), (0,1), (1,0), and (1,1)
	axs[int(i/plotWidth), i%plotWidth].scatter([0,1],[0,1], color='k', 
		marker='o', s=50)
	axs[int(i/plotWidth), i%plotWidth].scatter([0,1],[1,0], color='w', 
		marker='o', s=50)

# Remove all the unused axes
for i in range(len(epochList), plotWidth*plotWidth + remainder + 1):
	fig.delaxes(axs[int(i/plotWidth), i%plotWidth])

fig.set_size_inches(min(5*plotWidth, 20), 
	min(5*(remainder+plotWidth), 20), forward=True)
plt.show()



# # Run the untrained neural network on the model and colour outputs according to
# # a scale with blue being 0 and red being 1
# Y = model.predict(X)
# c = [(y[0], 0, y[1]) for y in Y]
# # c = ['b' if y[0]>0.5 else 'r' for y in Y]
# Xt = np.transpose(X)

# fig.set_size_inches(5,15, forward=True)
# axs[0].scatter(Xt[0], Xt[1], color=c)

# # Train for a little while to view progress
# intermediateEpochs = 500
# model.fit(x, y, batch_size=4, nb_epoch=intermediateEpochs, verbose=0)

# # Run the partly trained neural network to view progress
# Y = model.predict(X)
# c = [(y[0], 0, y[1]) for y in Y]

# # Visualise the output
# Xt = np.transpose(X)
# axs[1].scatter(Xt[0], Xt[1], color=c)


# maxEpochs = 1000
# model.fit(x, y, batch_size=4, nb_epoch=maxEpochs-intermediateEpochs, verbose=0)

# # Run the trained neural network and recolour points
# Y = model.predict(X)
# c = [(y[0], 0, y[1]) for y in Y]

# print(model.predict(x))

# # Visualise the output
# Xt = np.transpose(X)
# axs[2].scatter(Xt[0], Xt[1], color=c)

# plt.show()