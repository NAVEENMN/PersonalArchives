import os
import glob
import argparse
import shutil

home_dir = "/home/ubuntu"
arributes_directory = os.path.join(home_dir, "temp/raw_images", "annotations", "xmls")
images_directory = os.path.join(home_dir, "temp/raw_images", "images")
train_dir = os.path.join(home_dir, "data", "images", "train")
test_dir = os.path.join(home_dir, "data", "images", "test")

parser = argparse.ArgumentParser()
parser.add_argument('--attr_path', default=arributes_directory)
parser.add_argument('--images_path', default=images_directory)
parser.add_argument('--attr_format', default="xml")
parser.add_argument('--images_format', default="jpg")
args = parser.parse_args()

if args.attr_format == "xml":
  all_attributes_files = glob.glob(args.attr_path+'/*.xml')

# for each annotation file check if raw image exists
attr_files = []
look_up = dict()
for xml_file_path in all_attributes_files:
    image_name = xml_file_path.split(".")[0]
    image_name = image_name.split("/")[-1]
    look_up[xml_file_path] = image_name

for file_names in look_up.keys():
    image_path = os.path.join(images_directory, look_up[file_names]+".jpg")
    # check if this file exists
    if os.path.exists(image_path):
        look_up[file_names] = image_path
    else:
        look_up[file_names] = None

# split data in 80 20
total_data = len(look_up)
test_size = int(len(look_up) * 0.2)
train_size = total_data - test_size

# move data to train and test dir
indx = 0
for attr_file in look_up.keys():
    atr_src, img_src = attr_file, look_up[attr_file]
    if indx < train_size:
        # move to train dir
        target_dir = train_dir
        shutil.move(atr_src, train_dir)
        shutil.move(img_src, train_dir)
    else:
        #move to test dir
        target_dir = test_dir
        shutil.move(atr_src, test_dir)
        shutil.move(img_src, test_dir)
    print("{} --> {}".format(atr_src, target_dir))
    print("{} --> {}".format(img_src, target_dir))
    indx += 1
