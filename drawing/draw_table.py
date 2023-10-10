import os, sys, getopt
import json
from matplotlib import pyplot as plt
import ast


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
    
    print(my_arr)
            
    '''
    draw plot
    '''
    plt.figure(figsize=(16, 4))
    plt.plot(my_arr, linestyle=':', marker='o', color='tab:blue')

    #plt.xlim([0,max(train_it)+1])
    plt.ylim([0.1,0.135])
    plt.grid()
    #plt.legend()
    #plt.xlabel(x_label)
    #plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_plot_filename)
    plt.close()

if __name__ == "__main__":
    main()
