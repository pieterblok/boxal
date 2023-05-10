import os, glob
import json, csv

"""
0. get the inpput parameters
1. read csv file and store in a list
2. iterate over files in the png directory
3. check which row of the csv file corresponds with the png file in the train dir
4. grab the information about x bounderies of that row
5. generate a dictionary for each file
6. create a labelme json file for each png and write the dictionary to it
"""

'''
0. input parameters:
the  annotate subdir inside the train dir is autmatically generated while training and the new annotation should be placed there.
'''
n = 200 # number of rows to be read from the csv file
csv_filename = "/projects/parisa/data/progstar/box_annot/box_annotation_final.csv"
png_dir = "/projects/parisa/data/test_boxal/faster_rcnn/train/annotate/"
json_dir = "/projects/parisa/data/test_boxal/faster_rcnn/train/annotate/"


'''
1. read csv file and store it in an array -> rows
TODO: check if the memory capacity is enough for storing the array 
'''
fileds=[]
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

'''
2. iterate over png files and make a dictionary for each
then dump them into the json outfile
'''
for png_file in glob.iglob(os.path.join(png_dir, '*.png')): 
    height = int(os.path.basename(png_file).split('_')[5].split('z')[1])
    width = int(os.path.basename(png_file).split('_')[6].split('x')[1])
    print("processing", os.path.basename(png_file)[:-4])
    annotation_exits = False
    '''
    3. find the corresponding row of the csv
    '''        
    for row in rows:
        if row[0] == os.path.basename(png_file):
            annotation_exits = True
            '''
            4,5. extract box annotation from each row and generate a dictionary for each file
            '''
            writedata = {}
            writedata["version"] = "4.5.6"
            writedata["flags"] = {}
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
            print(shapes)
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

    if annotation_exits == False:
        print("annotation not found!")

