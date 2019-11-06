import os

anns_dir = 'data/train/anns/'
imgs_dir = 'data/train/imgs/'

folders = 'anns'

anns = os.listdir(anns_dir)

for filename in anns:

	print(filename)

	file_str = None

	with open(anns_dir + filename, 'r') as file:
		file_str = file.read()

		folder_start_idx = file_str.index('<folder>') + len('<folder>')
		folder_end_idx = file_str.index('</folder>')
		file_str = file_str[:folder_start_idx] + folders + file_str[folder_end_idx:]

		filename_start_idx = file_str.index('<filename>') + len('<filename>')
		filename_end_idx = file_str.index('</filename>') - 4
		file_str = file_str[:filename_start_idx] + filename.split('.')[0] + file_str[filename_end_idx:]

		path_start_idx = file_str.index('<path>') + len('<path>')
		path_end_idx = file_str.index('</path>')
		file_str = file_str[:path_start_idx] + imgs_dir + file_str[path_end_idx:]

	with open(anns_dir + filename, 'w') as file:
		file.write(file_str)

