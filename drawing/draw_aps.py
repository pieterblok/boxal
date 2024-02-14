import os, sys, getopt
import json
from matplotlib import pyplot as plt
import ast
import numpy as np

def main():
    '''
    input agruments
    '''
    output_plot_filename = "infer_APs.png"
    y_limit = 0

#    aps = [46.8, 51.2, 51, 54.3, 55.8, 54.2, 57.8, 58.4, 56.2]
#    ap50s = [78.5, 83.3, 78.6, 81.6, 82, 82.6, 83.1, 83.5, 81.7]
#    ap75s = [53.4, 56.7, 60.3, 63.8, 67.3, 61.9, 67.5, 66.2, 63.3]
#    aps = [43.1,43.3,47.9,48.6,48]
#    ap50s = [75.9,80.4,81.1,80,80]
#    ap75s = [43.3,44.3,57,50.6,50.3]               
#   aps = [43.1,43.3,47.9,48.6,48,46.8, 51.2, 51, 54.3, 55.8, 54.2, 57.8, 58.4, 56.2]
#    ap50s = [75.9,80.4,81.1,80,80,78.5, 83.3, 78.6, 81.6, 82, 82.6, 83.1, 83.5, 81.7]
#    ap75s = [43.3,44.3,57,50.6,50.3,53.4, 56.7, 60.3, 63.8, 67.3, 61.9, 67.5, 66.2, 63.3]               
#    aps = [43.1,43.3,47.9,48.6,48,48.7,49.1,48.9,47.2,47.3,48.5,49.1,49.2,49.2,59.4]
#    ap50s = [75.9,80.4,81.1,80,80,81.1,81.5,81.4,79.2,80.9,81.2,81.2,81.1,81.1,93.2]
#    ap75s = [43.3,44.3,57,50.6,50.3,51.6,54.5,54.4,53,53,50.9,50.9,52.7,52.8,63.7]
    aps = [41.5,53.7,57.4,59.6,62,63.1,62.7,63.9,64.87,65.31]
    ap50s = [81.9,87.6,90.5,91.2,90.6,91.5,89.4,91.5,91.71,90.55]
    ap75s = [36.9,63.1,68.6,71.4,76.4,74.2,73.7,75.7,76.31,76.53]
    x_idx = np.arange(1, len(aps) + 1)

    '''
    draw AP plots
    '''
    plt.figure(figsize=(15, 5)) #(16,7)
    if aps!=[]:
        plt.plot(x_idx, aps, label = 'mAP', linestyle=':', marker='o', color='tab:blue')
        plt.plot(x_idx, ap50s, label = 'AP50', linestyle=':', marker='o', color='tab:red')
        plt.plot(x_idx, ap75s, label = 'AP75', linestyle=':', marker='o', color='tab:green')

    plt.xlim([0.5, 10.5])
    plt.xticks(np.linspace(1, 10, 10))
    plt.ylim([35,96])
    plt.grid()
    plt.legend()
    plt.xlabel("Loops")
    plt.ylabel("AP values")
    plt.tight_layout()
    plt.savefig(output_plot_filename)
    plt.close()

if __name__ == "__main__":
    main()
