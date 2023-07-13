import os, sys, getopt
import json
from matplotlib import pyplot as plt


def main(argv):
    '''
    input agruments
    '''
    metrics_filename = ""
    output_plot_filename = ""
    plot_all = False

    opts, args = getopt.getopt(argv,"hi:o:n",["input_filename=","output_plot_filename="])
    for opt, arg in opts:
        if opt == '-h':
            print('python draw_metrics.py -i <input_filename> -o <output_plot_filename> -p\n \
                -i (--input_filename): input_filename is the path to the metrics.json file\n \
                -o (--output_plot_filename): output_plot_filename is the path to the name of png file to save the plots\n \
                -p (--plot all): plot all other losses other than the total loss. (Default=False)')
            sys.exit()
        elif opt in ("-i", "--input_filename"):
            metrics_filename = arg
        elif opt in ("-o", "--output_plot_filename"):
            output_plot_filename = arg
        elif opt in ("-p", "--plot_all"):
            plot_all = True


    if len(argv) < 3:
        sys.exit('ERROR: input arguments not provided!\n \
            python draw_metrics.py -i <input_filename> -o <output_plot_filename> -p\n \
                -i (--input_filename): input_filename is the path to the metrics.json file\n \
                -o (--output_plot_filename): output_plot_filename is the path to the name of png file to save the plots\n \
                -p (--plot all) - optional: plot all other losses other than the total loss. (Default=False)')

    if not os.path.exists(metrics_filename):
        sys.exit("ERROR: input file does not exist! {}".format(metrics_filename))


    '''
    read metrics from the file and save them into arrays
    '''
    train_it = [] # training iteration number
    val_it = []
    loss_tot = []
    loss_rpn_cls = []
    loss_rpn_loc = []
    loss_box_reg = []
    loss_cls = []
    accu = []
    loss_sum = []
    loss_val = []
    with open(metrics_filename) as f:
        for json_obj in f:
            dict = json.loads(json_obj)
            if "data_time" in dict:
                train_it.append(dict['iteration'])
                loss_tot.append(dict['total_loss'])
                accu.append(dict['fast_rcnn/cls_accuracy'])
                loss_rpn_cls.append(dict['loss_rpn_cls'])
                loss_rpn_loc.append(dict['loss_rpn_loc'])
                loss_box_reg.append(dict['loss_box_reg'])
                loss_cls.append(dict['loss_cls'])
                if 'validation_loss' in dict:
                    loss_val.append(dict['validation_loss'])
                    val_it.append(dict['iteration'])
                s = dict['loss_rpn_cls'] + dict['loss_rpn_loc'] + dict['loss_box_reg'] + dict['loss_cls']
                loss_sum.append(s)

    accumulate_it = 0
    for i, it in enumerate(train_it):
        if i+1 <len(train_it):
            if train_it[i] > train_it[i+1]:
                accumulate_it += train_it[i]+1
        train_it[i] = it + accumulate_it

    accumulate_it = 0
    for i, it in enumerate(val_it):
        if i+1 <len(val_it):
            if val_it[i] > val_it[i+1]:
                accumulate_it += val_it[i]+1
        val_it[i] = it + accumulate_it


    '''
    - sort arrays to get nice plots
    - assign repeated elements to zero, just to ignore them.
    - sort them once more to move zeroed elements to the begining
    '''
    train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum = zip(*sorted(zip(train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum)))
    if len(val_it)>0:
        val_it, loss_val = zip(*sorted(zip(val_it, loss_val)))


#    temp=-1
#    for idx,it in enumerate(train_it):
#        if it==temp:
#            train_it[idx-1]=0
#            loss_tot[idx-1]=0
#            loss_rpn_cls[idx-1]=0
#            loss_rpn_loc[idx-1]=0
#            loss_box_reg[idx-1]=0
#            loss_cls[idx-1]=0
#            accu[idx-1]=0
#        else:
#            temp = it
#    train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum = [list(t) for t in zip(*sorted(zip(train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum)))]

    '''
    draw plots
    '''
    plt.figure(figsize=(12, 7))
    if not plot_all:
        plt.plot(train_it, loss_tot, label = 'train', linestyle='-', color='tab:blue')
        if loss_val!=[]: plt.plot(val_it, loss_val, label = 'validation', linestyle='-', color='tab:orange')
    if plot_all:
        plt.plot(train_it, loss_tot, label = 'loss_total', marker = 'o', markersize=4, linestyle=':')
        plt.plot(train_it, loss_cls, label = 'loss_cls', marker = 's', markersize=4, linestyle=':')
        plt.plot(train_it, loss_box_reg, label = 'loss_box_reg', marker = '^', markersize=4, linestyle=':')
        plt.plot(train_it, loss_rpn_cls, label = 'loss_rpn_cls', marker = 'x', markersize=4, linestyle='none', fillstyle='none')
        plt.plot(train_it, loss_rpn_loc, label = 'loss_rpn_loc', marker = '+', markersize=4, linestyle='none', fillstyle='none')
    #plt.plot(train_it, loss_sum, label = 'loss_sum', marker = 'o', markersize=3, linestyle='none')
    #plt.plot(train_it, accu, marker = 'o', markersize=3, linestyle='none', fillstyle='none')

    plt.xlim([0,max(train_it)+1])
    plt.ylim([0,0.4])
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss values")
    plt.tight_layout()
    plt.savefig(output_plot_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
