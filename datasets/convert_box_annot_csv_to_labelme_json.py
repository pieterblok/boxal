import os, glob, sys, getopt
import json, csv

'''
usage: python convert_box_annot_csv_to_labelme_json.py -n <number_of_rows_to_be_read_from_csv_file> -c <csv_filename> -p <png_dir> -j <json_dir>'
'''

"""
0. get the inpput parameters
1. read csv file and store in a list
2. iterate over files in the png directory
3. check which row of the csv file corresponds with the png file in the train dir
4. grab the information about x bounderies of that row
5. generate a dictionary for each file
6. create a labelme json file for each png and write the dictionary to it
"""

def read_csv_file(csv_filename, n):
    '''
    1. read csv file and store it in an array -> rows
    TODO: check if the memory capacity is enough for storing the array 
    '''
    rows=[]
    c=0
    with open(csv_filename, 'r') as csv_f:
        csv_reader = csv.reader(csv_f)
        fields = next(csv_reader) # The line will get the first row of the csv file (Header row)
        c+=1
        for row in csv_reader:
            if c<n: #read the first n line only
                rows.append(row)
                c+=1
    return rows

def create_annotation_dictionary_from_row(row):
    box_num = 0
    shapes = []
    while (box_num+1)*4<len(row) and not row[(box_num+1)*4]=='': #the first x1,x2,y1,y2 starts from row[4] (accidentally). the next box is the next 4 columns and so on till there is void ''
        x_box = [float(row[(box_num+1)*4]), float(row[(box_num+1)*4+1])]
        y_box = [float(row[(box_num+1)*4+2]), float(row[(box_num+1)*4+3])]
        points = [(x_box[0], y_box[0]), (x_box[1], y_box[1])] 
        shape = {
            "label": "damaged",
            "line_color": None,
            "fill_color": None,
            "points": points,
            "shape_type": "rectangle",
            "group_id": None,
            "flags": {}
        }
        shapes.append(shape)
        box_num += 1
    return shapes
    
def write_labelme_json_file(row, png_file, json_dir):
    '''
    4,5. extract box annotation from each row and generate a dictionary for each file
    '''
    height = int(os.path.basename(png_file).split('_')[5].split('z')[1])
    width = int(os.path.basename(png_file).split('_')[6].split('x')[1])
    writedata = {}
    writedata["version"] = "4.5.6"
    writedata["flags"] = {}
    shapes = create_annotation_dictionary_from_row(row)
    writedata["shapes"] = shapes
    writedata["imagePath"]= png_file
    writedata["imageData"] = None
    writedata["imageHeight"] = height
    writedata["imageWidth"] = width

    '''
    6. create the json file and dump the annotation into it
    '''
    output_json_file = os.path.basename(png_file)[:-4] + ".json"
    with open(os.path.join(json_dir, output_json_file), 'w') as outfile:
        json.dump(writedata, outfile)

def main(argv):
    '''
    0. input parameters:
    '''
    n = 1000 # number of rows to be read from the csv file
    csv_filename = "/projects/parisa/data/progstar/box_annot/box_annotation_final.csv"
    png_dir = "/projects/parisa/data/test_boxal/faster_rcnn/train/annotate/"
    json_dir = "/projects/parisa/data/test_boxal/faster_rcnn/train/annotate/"

    opts, args = getopt.getopt(argv,"hn:c:p:j:",["num_rows=","csv_file=","png_dir=","json_dir="])
    for opt, arg in opts:
        if opt == '-h':
            print ('python convert_box_annot_csv_to_labelme_json.py -n <number_of_rows_to_be_read_from_csv_file> -c <csv_filename> -p <png_dir> -j <json_dir>')
            sys.exit()
        elif opt in ("-n", "--num_rows"):
            n = int(arg)
        elif opt in ("-c", "--csv_file"):
            csv_filename = arg
        elif opt in ("-p", "--png_dir"):
            png_dir = arg
        elif opt in ("-j", "--json_dir"):
            json_dir = arg
    if len(argv) < 5:
        print("WARNING: default input files will be used!")

    if not os.path.exists(csv_filename):
        sys.exit("ERROR: {} does not exist!".format(csv_filename))
    if not os.path.exists(png_dir):
        sys.exit("ERROR: {} does not exist!".format(png_dir))
    if not os.path.exists(json_dir):
        sys.exit("ERROR: {} does not exist!".format(json_dir))
    
    rows = read_csv_file(csv_filename, n)
    
    '''
    2. iterate over png files and make a dictionary for each
    then dump them into the json outfile
    '''
    for png_file in glob.iglob(os.path.join(png_dir, '*.png')): 
        print("processing", os.path.basename(png_file)[:-4])
        annotation_exits = False
        '''
        3. find the corresponding row of the csv and create json for it
        '''       
        for row in rows:
            if row[0] == os.path.basename(png_file):
                annotation_exits = True
                write_labelme_json_file(row, png_file, json_dir)

        if annotation_exits == False:
            print("annotation not found!")

if __name__ == "__main__":
    main(sys.argv[1:])
