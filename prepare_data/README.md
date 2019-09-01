# Prepare data

The primary goal of this module is to generate tfrecords files. The TFRecord format is a simple format for storing a sequence of binary records. Tensorflow uses this format of data since is space efficient and faster.

## Prerequisite

Before begining this section its very important to understand how our data is structured. Different data comes in different structures and configrations and hence we needto to build custom scripts to cater that needs. In this case our data means raw images and assosiated annotation files. annotations files hold information like object classes, pixel co ordinates etc. typicall annotation files comes in xml format along with raw images.

If data comes in tfrecords then its great!!. If it comes in csv format then we need to run
```
python generate_tfrecord.py --csv_input <path> --image_dir <path> --output_path <path>
``` 
to generate tfrecord files.

If data comes in xml format we need to run 
```
xml_to_csv.py
```
to convert annotation files to csv format.

## arranging the data

Data can be stored in S3 or downloaded locally. We need to dump the data in correct directories so we can exectue above commands. for that purpose 
```
move_data.py
```
is introduced.
