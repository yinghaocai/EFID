'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: Create single.csv for tagert images
usage0: python create_single_csv.py --pdf_dir dir_of_pdf --title title_of_paper --kind kind_of_img
usage1: python create_single_csv.py --pdf_dir dir_of_pdf --title title_of_paper --kind kind_of_img --DOI DOI_seq
'''

# import necessary packages
from imutils import paths
import pandas as pd
import argparse
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
  ap.add_argument("-p", "--pdf_dir", required=True,
                  help="Require the directory containing the pdf")
  ap.add_argument("-k", "--kind", required=True,
                  help="Require the kind of images.")
  ap.add_argument("-t", "--title", required=True,
                  help="Require title of the paper.")
  ap.add_argument("-u", "--URL", required=True,
                  help="Require URL to downloading the file.")
  ap.add_argument("-d", "--DOI", default='',
                  help="Require DOI if it exists.")
  args = vars(ap.parse_args())

  # initial basic parameters
  images_dir = os.path.join(args['pdf_dir'], args['kind'])
  title = args['title']
  DOI = args['DOI']
  URL = args['URL']

  # create single.csv
  csv_path = os.path.join(args['pdf_dir'],'single_%s.csv' % args['kind'])
  csv_head = ['filename', 'title', 'DOI', 'URL']
  create_csv(csv_path, csv_head)

  # populate single.csv
  images = paths.list_images(images_dir)
  for image in images:
    filename = os.path.split(image)[-1]
    csv_data = [filename, title, DOI, URL]
    write_csv(csv_path, csv_data)

