'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: legalize the image filename, and create processed-label directory
usage: python prepare_smoother.py --dataset Pdfs
'''

from imutils.paths import list_images
import argparse
import shutil
import os


def legalize(s):
  s_new = ''
  for c in s:
    if c > chr(127):
      continue
    s_new += c
    
  s_len = len(s_new)
  if s_len > 110:
    return s_new[:100]+s_new[s_len -10:]
  return s_new

def mkdir_if_not_exists(path):
  if not os.path.exists(path):
    os.makedirs(path)


if __name__ == '__main__':
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-d", "--dataset", required=True,
                  help="the directory of dataset.")
  args = vars(ap.parse_args())
  dataset=args['dataset']
  if not os.path.exists(dataset):
    print('No "%s" directory exists!'%args['dataset'])
    exit()
  
  images=list(list_images(dataset))
  for image in images:
    # legalize all paths of images
    p, f = os.path.split(image)
    f = legalize(f)
    os.rename(image, os.path.join(p,f))
    
    # locate the label image path
    if os.path.split(p)[-1] != 'label':
      continue
      
    # create the processed-label directory
    processed = os.path.join(os.path.split(p)[0], 'processed-label')
    mkdir_if_not_exists(processed)
      
    # rename the picture
    n, e = os.path.splitext(f)
    if n.find('.') == -1:
      continue
    n = n.replace('.', '_')
    image_new = os.path.join(p, n+e)
    os.rename(image, image_new)
