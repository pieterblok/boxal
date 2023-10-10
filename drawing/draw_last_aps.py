import os, sys, getopt
import json
from matplotlib import pyplot as plt
import ast

def find_last_match(pattern, filename):
    last_matching_line = None
    with open(filename, 'r') as file:
        for line in file:
            if pattern in line:
                last_matching_line = line
    
    if last_matching_line:
        return last_matching_line.strip()  # Remove trailing newline if needed
    else:
        print("Pattern not found in the file.", filename)

def main():
    '''
    input agruments
    '''
    input_args = input("enter the metric file names in the order you want to plot, separated by space and press enter:\n")
    metrics_filenames = input_args.split()
    output_plot_filename_prefix = "last_APs"
    plot_all = True
    y_limit = 0

    aps = []
    ap50s = []
    ap75s = []
    for metrics_filename in metrics_filenames:
        if not os.path.exists(metrics_filename):
            sys.exit("ERROR: input file does not exist! {}".format(metrics_filename))
        else:
            metrics_line = find_last_match("AP50", metrics_filename)
            metrics_dict = ast.literal_eval(metrics_line.replace("NaN", "None"))
            aps.append(metrics_dict["bbox/AP"])
            ap50s.append(metrics_dict["bbox/AP50"])
            ap75s.append(metrics_dict["bbox/AP75"])
            
    '''
    draw AP plots
    '''
    plt.figure(figsize=(10, 7))
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
    plt.savefig(output_plot_filename_prefix+".png")
    plt.close()

if __name__ == "__main__":
    main()
