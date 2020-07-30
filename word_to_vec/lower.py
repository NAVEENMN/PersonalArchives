import os
import sys
import re
from itertools import chain
from glob import glob

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_path = os.path.join(current_path, 'data')

for filename in os.listdir(data_path):
	print(filename)
	if filename.endswith(".txt"):
		file_path = os.path.join(data_path, filename)
		f = open(file_path, 'r', encoding='utf-8-sig')
		text = f.read()
		text = text.lower()
		text = re.sub(r'[^a-zA-Z0-9_\s]+', '', text)
		with open(file_path, 'w') as out:
			out.writelines(text)