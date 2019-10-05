import sys
import os
import json
from json import dumps
from xml.etree.ElementTree import fromstring
from xmljson import badgerfish as bf

# Generate your own annotation file and class names file.
# One row for one image;
# Row format: image_file_path box1 box2 ... boxN;
# Box format: x_min,y_min,x_max,y_max,class_id (no space).
# For VOC dataset, try python voc_annotation.py
# Here is an example:

# path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
# path/to/img2.jpg 120,300,250,600,2


def check_image_file_exist(filename):
	image_file_type = ['.bmp', '.jpg', '.jpeg', '.png', '.gif']
	filename = os.path.splitext(filename)[0]
	
	for file_type in image_file_type:
		if os.path.isfile(filename + file_type):
			return (filename + file_type)

	return None

def get_obj_info(obj):
	class_name = obj['name']['$']
	bndbox = obj['bndbox']
	x_min = bndbox['xmin']['$']
	x_max = bndbox['xmax']['$']
	y_min = bndbox['ymin']['$']
	y_max = bndbox['ymax']['$']
	return (x_min, y_min, x_max, y_max, class_name)



# python convert_to_xml.py obj.names directory/
names_filename = sys.argv[1]
class_arr = {}
class_id = 0

with open(names_filename, 'r') as names_file:
	line = names_file.readline().rstrip('\n')
	while line:
		class_arr[line] = class_id
		line = names_file.readline().rstrip('\n')
		class_id += 1



directory = sys.argv[2]

tmp = ''

for filename in os.listdir(directory):
	filename = os.path.join(directory, filename)
	if filename.endswith('.xml'):
		image_filename = check_image_file_exist(filename)
		if image_filename != None:
			tmp += image_filename + ' '
		else:
			print('Image file not found.')

		with open(filename, 'r') as file:
			data = json.loads(dumps(bf.data(fromstring(file.read()))))['annotation']
			obj_arr = data['object']

			if type(obj_arr) != type([]):
				obj_arr = [obj_arr]

			for obj in obj_arr:
				(x_min, y_min, x_max, y_max, class_name) = get_obj_info(obj)
				tmp += str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(class_arr[class_name]) + ' '

			tmp += '\n'

print(tmp)

