'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: download the image according to the keyword(Chinese) and engine
usage: python main.py --keyword "" --engine "" --output "Images"
'''

# import the necessary packages
from search_in_sougou import save_images as sougou
from search_in_baidu import save_images as baidu
from search_in_bing import save_images as bing
from common import convert_images
import argparse

if __name__=='__main__':
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-e", "--engine", required=True,
      help="the choose of searching engine among baidu, sougou and bing")
  ap.add_argument("-k", "--keyword", required=True,
      help="the keyword for what you are searching")
  ap.add_argument("-o", "--output", required=True,
      help="path to saving the images")
  ap.add_argument("-p", "--pages", type=int, default=20,
      help="Number of pages")
  ap.add_argument("-t", "--timeout", type=float, default=3.04,
      help="Waiting time for each request")
  args = vars(ap.parse_args())
  
  # save images from designated search engine
  if(args['engine'].lower() == 'baidu'):
    baidu(keyword = args['keyword'], path_save = args['output'], pages = args['pages'], timeout = args['timeout'])
  elif (args['engine'].lower() == 'sougou'):
    sougou(keyword = args['keyword'], path_save = args['output'], pages = args['pages'], timeout = args['timeout'])
  elif (args['engine'].lower() == 'bing'):
    bing(keyword = args['keyword'], path_save = args['output'], pages = args['pages'], timeout = args['timeout'])
  else:
    print('This crawler is only available for Baidu, Sohu and Bing')
  
  # convert images from the output path
  convert_images(args['output'])
