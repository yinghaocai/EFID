'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: Merge all single.csv created by running create_single_csv.py
usage0: python merge_single_csv.py --pdfs Pdfs --kind non-label --output ../Processed
usage1: python merge_single_csv.py --pdfs Pdfs --kind processed-label --output ../Processed
'''

# import necessary packages
from imutils import paths
import argparse
import shutil
import csv
import os


def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_csv(csv_path, csv_head):
  with open(csv_path, 'w') as f:
    csv_write = csv.writer(f)
    csv_write.writerow(csv_head)


def write_csv(csv_path, csv_data):
  with open(csv_path, 'a+') as f:
    csv_write = csv.writer(f, dialect = 'excel')
    csv_write.writerow(csv_data)


def merge_csv(csv_target, csv_src):
  with open(csv_src, 'r', encoding='UTF-8') as f:
    rows = list(csv.reader(f))
    for i in range(1, len(rows)):
      write_csv(csv_target, rows[i])


if __name__ == '__main__':

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--pdfs", required=True,
                  help="path to directory of saved pdfs")
  ap.add_argument("-k", "--kind", required=True,
                  help="the kind of single.csv")
  ap.add_argument('-o', '--output', default='./',
                  help='path to save merged.csv')
  args = vars(ap.parse_args())

  # initial basic parameters
  mkdir_if_not_exists(args['output'])
  merge_path = os.path.join(args['output'], 'metadata_%s.csv' % args['kind'])
  pdfs_dir = args['pdfs']
  pdf_dirs = os.listdir(pdfs_dir)

  # create metadata.csv
  csv_head = ['filename', 'title', 'DOI', 'URL']
  create_csv(merge_path, csv_head)

  for i in range(len(pdf_dirs)):
    if not os.path.isdir(os.path.join(pdfs_dir, pdf_dirs[i])):
      continue

    # merge the single.csv to metadata.csv
    csv_src = os.path.join(pdfs_dir, pdf_dirs[i], 'single_%s.csv' % args['kind'])
    merge_csv(merge_path, csv_src)

  # copy all images in imgs_dirs to imgs_dir
  target_image_dir = os.path.join(args['output'], 'images_%s' % args['kind'])
  mkdir_if_not_exists(target_image_dir)
  src_image_dirs = [os.path.join(pdfs_dir, pdf_dir, args['kind']) for pdf_dir in pdf_dirs]
  for src_image_dir in src_image_dirs:
    for src_image_path in paths.list_images(src_image_dir):
      shutil.copy(src_image_path, target_image_dir)
