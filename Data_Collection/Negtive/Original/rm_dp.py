'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: rm the duplication iamge to reduce manual workload
usage: python rm_dp.py --input "Images" --csv "checked.csv"
'''

# import the necessary packages
from imutils import paths
import pandas as pd
import argparse
import os


if __name__ == '__main__':

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--input", required=True,
                  help="path to remove the duplication according the filename")
  ap.add_argument('-c', '--csv', required=True,
                  help='path to the file.csv')
  args = vars(ap.parse_args())

  # initial basic parameters
  if not os.path.exists(args['input']):
    print('the path to directory of images not exists.')
    exit()
    
  # check
  check_list = list(pd.read_csv(args["csv"])['filename'])
  images=list(paths.list_images(args['input']))
  for image in images:
    filename = os.path.split(image)[-1]
    if os.path.splitext(filename)[0] in check_list:
      os.remove(image)

