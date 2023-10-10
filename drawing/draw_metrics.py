import os, sys, getopt
import json
from matplotlib import pyplot as plt


def main(argv):
    '''
    input agruments
    '''
    metrics_filename = ""
    output_plot_filename_prefix = ""
    plot_all = False
    y_limit = 0

    opts, args = getopt.getopt(argv,"hi:o:y:p",["input_filename=","output_plot_filename_prefix=", "y_limit="])
    for opt, arg in opts:
        if opt == '-h':
            print('python draw_metrics.py -i <input_filename> -o <output_plot_filename> -p\n \
                -i (--input_filename): input_filename is the path to the metrics.json file\n \
                -o (--output_plot_filename): output_plot_filename is the path to the name of png file to save the plots\n \
                -p (--plot_all): plot all other losses other than the total loss. (Default=False)\n \
                -y (--y_limit): limit on the maximum of y axis for the loss value')
            sys.exit()
        elif opt in ("-i", "--input_filename"):
            metrics_filename = arg
        elif opt in ("-o", "--output_plot_filename_prefix"):
            output_plot_filename_prefix = arg
        elif opt in ("-y", "--y_limit"):
            y_limit = float(arg)
        elif opt in ("-p", "--plot_all"):
            plot_all = True


    if len(argv) < 3:
        sys.exit('ERROR: input arguments not provided!\n \
            python draw_metrics.py -i <input_filename> -o <output_plot_filename> -p\n \
                -i (--input_filename): input_filename is the path to the metrics.json file\n \
                -o (--output_plot_filename_prefix): is the path + prefix to the name of png file to save the plots\n \
                -p (--plot all) - optional: plot all other losses other than the total loss. (Default=False)\n \
                -y (--y_limit): limit on the maximum of y axis for the loss value')

    if not os.path.exists(metrics_filename):
        sys.exit("ERROR: input file does not exist! {}".format(metrics_filename))


    '''
    read metrics from the file and save them into arrays
    '''
    train_it = [] # training iteration number
    val_it = []
    ap_it = []
    loss_tot = []
    loss_rpn_cls = []
    loss_rpn_loc = []
    loss_box_reg = []
    loss_cls = []
    accu = []
    loss_sum = []
    loss_val = []
    ap = []
    ap50 = []
    ap75 = []
    loss_val_min = 100
    with open(metrics_filename) as f:
        for json_obj in f:
            dict = json.loads(json_obj)
            if "data_time" in dict:
                train_it.append(dict['iteration'])
                loss_tot.append(dict['total_loss'])
#                accu.append(dict['fast_rcnn/cls_accuracy'])
                loss_rpn_cls.append(dict['loss_rpn_cls'])
                loss_rpn_loc.append(dict['loss_rpn_loc'])
                loss_box_reg.append(dict['loss_box_reg'])
                loss_cls.append(dict['loss_cls'])
                if 'validation_loss' in dict:
                    loss_val.append(dict['validation_loss'])
                    val_it.append(dict['iteration'])
                    if dict['validation_loss']<loss_val_min: loss_val_min = dict['validation_loss']
                if "bbox/AP" in dict:
                    ap.append(dict["bbox/AP"])
                    ap50.append(dict["bbox/AP50"])
                    ap75.append(dict["bbox/AP75"])
                    ap_it.append(dict['iteration'])
                s = dict['loss_rpn_cls'] + dict['loss_rpn_loc'] + dict['loss_box_reg'] + dict['loss_cls']
                loss_sum.append(s)
    print("loss_val_min: ", loss_val_min)
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

    accumulate_it = 0
    for i, it in enumerate(ap_it):
        if i+1 <len(ap_it):
            if ap_it[i] > ap_it[i+1]:
                accumulate_it += ap_it[i]+1
        ap_it[i] = it + accumulate_it

    '''
    - sort arrays to get nice plots
    - assign repeated elements to zero, just to ignore them.
    - sort them once more to move zeroed elements to the begining
    '''
#    train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum = zip(*sorted(zip(train_it, loss_tot, accu, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum)))
    train_it, loss_tot, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum = zip(*sorted(zip(train_it, loss_tot, loss_rpn_cls, loss_rpn_loc, loss_box_reg, loss_cls, loss_sum)))
    if len(val_it)>0:
        val_it, loss_val = zip(*sorted(zip(val_it, loss_val)))
    if len(ap_it)>0:
        ap_it, ap, ap50, ap75 = zip(*sorted(zip(ap_it, ap, ap50, ap75)))


    '''
    draw loss plots
    '''
    plt.figure(figsize=(10, 7))
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
    if y_limit>0: plt.ylim([0,y_limit])
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss values")
    plt.tight_layout()
    plt.savefig(output_plot_filename_prefix+"_loss.png")
    plt.close()
    
    '''
    draw AP plots
    '''
    plt.figure(figsize=(10, 7))
    if ap!=[]:
        plt.plot(ap_it, ap, label = 'mAP', linestyle='-', color='tab:blue')
        plt.plot(ap_it, ap50, label = 'AP50', linestyle='-', color='tab:red')
        plt.plot(ap_it, ap75, label = 'AP75', linestyle='-', color='tab:green')

    plt.xlim([0,max(train_it)+1])
    plt.ylim([0,102])
    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("AP values")
    plt.tight_layout()
    plt.savefig(output_plot_filename_prefix+"_ap.png")
    plt.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
