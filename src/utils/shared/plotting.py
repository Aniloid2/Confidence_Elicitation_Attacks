import numpy as np
import matplotlib.pyplot as plt
from ternary import TernaryAxesSubplot, figure

def ternary_plot(samples, alpha, file_path=None): 

    scale = 1

    # Create a matplotlib figure and axis
    fig, tax = figure(scale=scale)
    tax.set_title(f"Dirichlet Distribution Samples (α={alpha})", fontsize=20)


    

    # Plot the samples
    tax.scatter(samples, color='blue', alpha=0.3, marker='o')
    
    # Ternary plot boundaries and labels
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", linestyle='dotted')

    # Ticks and labels
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1)
    tax.left_axis_label("Positive", fontsize=15)
    tax.right_axis_label("Negative", fontsize=15)
    tax.bottom_axis_label("Null", fontsize=15)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()

    # Save the plot to a file if file_path is provided
    if file_path:
        plt.savefig(file_path+'.png')
        plt.savefig(file_path+'.pdf')
    # Display the plot
    plt.show()

def ternary_mean_plot(samples, alpha,mean_point, file_path=None): 

    scale = 1

    # Create a matplotlib figure and axis
    fig, tax = figure(scale=scale)
    fig.suptitle(f"Dirichlet Distribution \nSamples (α={alpha})", fontsize=20, y=1.05)

    # Plot the samples [(0.1,0.8,0.1), ... 1000]
    tax.scatter(samples, color='blue', alpha=0.3, marker='o',label="Samples")

    # Plot the empirical mean
    tax.scatter([mean_point], color='red', alpha=0.9, marker='X', s=40, label="Empirical Mean")


    # Ternary plot boundaries and labels
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", linestyle='dotted')

    # Ticks and labels
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1,offset=0.02, tick_formats="%.1f")
    tax.left_axis_label("Null", fontsize=15, offset=0.15)
    tax.right_axis_label("Positive", fontsize=15, offset=0.15)
    tax.bottom_axis_label("Negative", fontsize=15, offset=0.06)

    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    # Set the legend
    tax.legend()
    # Save the plot to a file if file_path is provided
    if file_path:
        plt.savefig(file_path+'.png', bbox_inches='tight')
        plt.savefig(file_path+'.pdf', bbox_inches='tight')
    # Display the plot
    plt.show()