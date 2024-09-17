'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: Create a file.csv containing all the filename of the specific directory
usage: python create_csv.py --input "" --output ""
'''

# import necessary packages
from imutils import paths
import argparse
import shutil
import csv
import os


def create_csv(csv_path, csv_head):
  with open(csv_path, 'w') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(csv_head)


def write_csv(csv_path, csv_data):
  with open(csv_path, 'a+') as f:
    csv_write = csv.writer(f, dialect = 'excel')
    csv_write.writerow(csv_data)


if __name__ == '__main__':

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--input", required=True,
                  help="path to recording the filename")
  ap.add_argument('-o', '--output', required=True,
                  help='path to save file.csv')
  args = vars(ap.parse_args())

  # initial basic parameters
  if not os.path.exists(args['input']):
    print('the path to directory of images not exists.')
    exit()
  csv_path = args['output']
  images=list(paths.list_images(args['input']))
  csv_head = ['filename']
  
  # operate the file.csv
  create_csv(csv_path, csv_head)
  for image in images:
    filename = os.path.split(image)[-1]
    csv_data = [os.path.splitext(filename)[0],]
    write_csv(csv_path, csv_data)
  
