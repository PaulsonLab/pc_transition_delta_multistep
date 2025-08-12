import matplotlib.pyplot as plt
import os
import shutil
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def plotPCbatch(pcArray1, pcArray2, pcArray3, show=True, save=False, name=None, fig_count=9, sizex=12, sizey=4):
    # Select the data from the arrays
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    pc3 = pcArray3[0:fig_count]

    # Create a figure with three rows and fig_count columns
    fig = plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count * 3):
        ax = fig.add_subplot(3, fig_count, i + 1, projection='3d')
        
        # Plot data in the first row
        if i < fig_count:
            ax.scatter(pc1[i, :, 0], pc1[i, :, 2], pc1[i, :, 1], c='b', marker='.', alpha=0.3, s=8)
        
        # Plot data in the second row
        elif i < 2 * fig_count:
            ax.scatter(pc2[i - fig_count, :, 0], pc2[i - fig_count, :, 2], pc2[i - fig_count, :, 1], c='r', marker='.', alpha=0.3, s=8)
        
        # Plot data in the third row
        else:
            ax.scatter(pc3[i - 2 * fig_count, :, 0], pc3[i - 2 * fig_count, :, 2], pc3[i - 2 * fig_count, :, 1], c='g', marker='.', alpha=0.3, s=8)

        # Hide the axis
        plt.axis('off')

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Save the figure if save is True
    if save:
        fig.savefig(name + '.png')
        plt.close(fig)

    # Show the figure
    if show:
        plt.show()
    else:
        return fig