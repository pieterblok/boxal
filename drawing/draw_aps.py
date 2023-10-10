import os, sys, getopt
import json
from matplotlib import pyplot as plt
import ast


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
    aps = [43.1,43.3,47.9,48.6,48,46.8, 51.2, 51, 54.3, 55.8, 54.2, 57.8, 58.4, 56.2]
    ap50s = [75.9,80.4,81.1,80,80,78.5, 83.3, 78.6, 81.6, 82, 82.6, 83.1, 83.5, 81.7]
    ap75s = [43.3,44.3,57,50.6,50.3,53.4, 56.7, 60.3, 63.8, 67.3, 61.9, 67.5, 66.2, 63.3]               


    '''
    draw AP plots
    '''
    plt.figure(figsize=(16, 7))
    if aps!=[]:
        plt.plot(aps, label = 'mAP', linestyle=':', marker='o', color='tab:blue')
        plt.plot(ap50s, label = 'AP50', linestyle=':', marker='o', color='tab:red')
        plt.plot(ap75s, label = 'AP75', linestyle=':', marker='o', color='tab:green')

    #plt.xlim([0,max(train_it)+1])
    plt.ylim([40,96])
    plt.grid()
    plt.legend()
    plt.xlabel("Loops")
    plt.ylabel("AP values")
    plt.tight_layout()
    plt.savefig(output_plot_filename)
    plt.close()

if __name__ == "__main__":
    main()
