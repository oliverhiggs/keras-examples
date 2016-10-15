import numpy as np
import matplotlib.pyplot as plt

# TODO: Convert this into n-classification visualisation??

def progress_plot(model, xTrn, yTrn, bottomLeft, topRight, epochList,
                  batchSize, numPoints=101, alpha=0.1, verbose=False):
    ''' A function to plot the progress of training a neural network for a
    binary classification problem'''

    X = [[x1, x2] for x1 in np.linspace(bottomLeft[0], topRight[0], numPoints)
         for x2 in np.linspace(bottomLeft[1], topRight[1], numPoints)]

    # The figure will have a width equal to ceiling(sqrt(number))
    plotWidth = int(np.sqrt(len(epochList)))
    if plotWidth**2 < len(epochList):
        plotWidth += 1

    # The figure has a height equal to the ceiling of the number of axes
    # divided by the width
    plotHeight = (len(epochList) - 1) // plotWidth + 1

    fig, axs = plt.subplots(plotHeight, plotWidth, squeeze=False)

    epochList.sort()

    trained = 0

    for i in range(0, len(epochList)):
        axs[int(i / plotWidth), i % plotWidth].set_xlim(bottomLeft[0],
                                                        topRight[0])
        axs[int(i / plotWidth), i % plotWidth].set_ylim(bottomLeft[1],
                                                        topRight[1])
        axs[int(i / plotWidth), i % plotWidth].set_title("{0:d} Epochs".
                                                         format(epochList[i]))

        # Train the network
        model.fit(xTrn, yTrn, batch_size=batchSize, 
                  nb_epoch=epochList[i] - trained, verbose=verbose)
        trained = epochList[i]

        # Run the trained neural network and colour points on a sliding scale
        # with blue for 0 and red for 1
        Y = model.predict(X)
        c = [(Yi[0], 0, Yi[1]) for Yi in Y]

        # Plot the output
        Xt = np.transpose(X)
        axs[int(i / plotWidth), i % plotWidth].scatter(Xt[0], Xt[1], color=c,
                                                       alpha=alpha)

        # Add marks at the training points
        for j in range(len(xTrn)):
            markColour = [yTrn[j][0], 0, yTrn[j][1]]
            axs[int(i / plotWidth), i % plotWidth].scatter(xTrn[j][0],
                                                           xTrn[j][1],
                                                           color=markColour,
                                                           marker='o',
                                                           s=50)

    # Remove all the unused axes
    for i in range(len(epochList), plotWidth * plotHeight):
        fig.delaxes(axs[int(i / plotWidth), i % plotWidth])

    fig.set_size_inches(min(5 * plotWidth, 20),
                        min(5 * plotHeight, 20),
                        forward=True)
    model.predict(xTrn)
    plt.show()
