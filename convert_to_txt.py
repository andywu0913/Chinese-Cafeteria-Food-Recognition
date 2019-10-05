import sys
import os
import json
from json import dumps
from xml.etree.ElementTree import fromstring
from xmljson import badgerfish as bf

def get_obj_info(obj):
	global total_height
	global total_width
	class_name = obj['name']['$']
	bndbox = obj['bndbox']
	width = float(bndbox['xmax']['$'] - bndbox['xmin']['$'])
	height = float(bndbox['ymax']['$'] - bndbox['ymin']['$'])
	x_center = float(width / 2) + bndbox['xmin']['$']
	y_center = float(height / 2) + bndbox['ymin']['$']

	width = round(width / total_width, 6)
	height = round(height / total_height, 6)
	x_center = round(x_center / total_width, 6)
	y_center = round(y_center / total_height, 6)

	return (class_name, width, height, x_center, y_center)

# read .names file
class_filename = sys.argv[1]
class_file = open(class_filename, 'r')
line = class_file.readline().rstrip('\n')
index = 0

class_arr = {}

while line:
	class_arr[line] = index
	line = class_file.readline().rstrip('\n')
	index += 1


# read XML file
filename = sys.argv[2]
file = open(filename, 'r')
data = file.read()
file.close()
data = json.loads(dumps(bf.data(fromstring(data))))['annotation']

total_width = data['size']['width']['$']
total_height = data['size']['height']['$']

filename = os.path.splitext(filename)[0]
file = open(filename + '.txt', 'w')

obj_arr = data['object']

if type(obj_arr) != type([]):
	obj_arr = [obj_arr]

for obj in obj_arr:
	(class_name, width, height, x_center, y_center) = get_obj_info(obj)

	file.write(str(class_arr[class_name]) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n')



