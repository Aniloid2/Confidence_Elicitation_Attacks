import numpy as np
import matplotlib.pyplot as plt
from ternary import TernaryAxesSubplot, figure

def ternary_plot(samples, alpha, file_path=None): 

    scale = 1

     
    fig, tax = figure(scale=scale)
    tax.set_title(f"Dirichlet Distribution Samples (α={alpha})", fontsize=20)


    

     
    tax.scatter(samples, color='blue', alpha=0.3, marker='o')
    
     
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", linestyle='dotted')

     
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1)
    tax.left_axis_label("Positive", fontsize=15)
    tax.right_axis_label("Negative", fontsize=15)
    tax.bottom_axis_label("Null", fontsize=15)

     
    tax.clear_matplotlib_ticks()

     
    if file_path:
        plt.savefig(file_path+'.png')
        plt.savefig(file_path+'.pdf')
     
    plt.show()

def ternary_mean_plot(samples, alpha,mean_point, file_path=None): 

    scale = 1

     
    fig, tax = figure(scale=scale)
    fig.suptitle(f"Dirichlet Distribution \nSamples (α={alpha})", fontsize=20, y=1.05)

     
    tax.scatter(samples, color='blue', alpha=0.3, marker='o',label="Samples")

     
    tax.scatter([mean_point], color='red', alpha=0.9, marker='X', s=40, label="Empirical Mean")


     
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="black", linestyle='dotted')

     
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1,offset=0.02, tick_formats="%.1f")
    tax.left_axis_label("Null", fontsize=15, offset=0.15)
    tax.right_axis_label("Positive", fontsize=15, offset=0.15)
    tax.bottom_axis_label("Negative", fontsize=15, offset=0.06)

     
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
     
    tax.legend()
     
    if file_path:
        plt.savefig(file_path+'.png', bbox_inches='tight')
        plt.savefig(file_path+'.pdf', bbox_inches='tight')
     
    plt.show()