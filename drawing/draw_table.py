import os, sys, getopt
import json
from matplotlib import pyplot as plt
import ast
import numpy as np

def main():
    '''
    input agruments
    '''
    input_args = input("enter the array you want to plot, separated by space and press enter:\n")
    input_arr = input_args.split()
    my_arr = [float(i) for i in input_arr]
    output_plot_filename = "my_plot.png"
    y_limit = 0
    x_label = ""
    y_label = ""
    x_idx = np.arange(1, len(my_arr) + 1)
            
    '''
    draw plot
    '''
    plt.figure(figsize=(18, 2))
    plt.plot(x_idx, my_arr, linestyle=':', marker='o', color='tab:orange')

    plt.xlim([0.5, 10.5])
    plt.ylim([0.1,0.18])
    plt.xticks(np.linspace(1, 10, 10))
    plt.grid()
    #plt.legend()
    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_plot_filename)
    plt.close()

if __name__ == "__main__":
    main()
