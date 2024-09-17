# -*- coding: UTF-8 -*-
# usage0
# python search_in_baidu.py --keyword 岩石 --output 'baidu_rock_images' --pages 20 --timeout 3.05
# usage1
# python search_in_baidu.py --keyword 石头 --output 'baidu_stone_images' --pages 50 --timeout 3.05

# import necessary packages
from common import mkdirs_if_not_exists, legitimize, save_img, convert_images
import argparse
import requests
import os


# save imgs from dynamic web
def save_images(keyword, path_save, pages, timeout):

  # default images per page
  images = 30

  # get the all the Page urls
  def getPages(keyword, pages):
    print('  Geting %s Pages\' url%s...' % (pages, 's' if pages > 1 else ''))
    params_all=[]
    for i in range(images,images*pages+images,images):
      params_all.append({
                    'tn': 'resultjson_com',
                    'ipn': 'rj',
                    'ct': 201326592,
                    'is': '',
                    'fp': 'result',
                    'queryWord': keyword,
                    'cl': 2,
                    'lm': -1,
                    'ie': 'utf-8',
                    'oe': 'utf-8',
                    'adpicid': '',
                    'st': -1,
                    'z': '',
                    'ic': 0,
                    'word': keyword,
                    's': '',
                    'se': '',
                    'tab': '',
                    'width': '',
                    'height': '',
                    'face': 0,
                    'istype': 2,
                    'qc': '',
                    'nc': 1,
                    'fr': '',
                    'pn': i,
                    'rn': 30,
                    'gsm': '1e',
                    '1488942260214': ''
                })
    url = 'https://image.baidu.com/search/acjson'
    urls = []
    i = 0
    for params in params_all:
      try:
        urls.append(requests.get(url,params=params).json().get('data'))
        i += 1
      except:
        break
    print('  Geted %d Page%s\' url.\n' % (i, 's' if i > 1 else ''))
    return urls

  mkdirs_if_not_exists(path_save)
  dataList = getPages(keyword, pages)  # param1:keyword must be Chinese
  m = len(dataList)
  n = len(dataList[0])
  
  for i in range(m):
    print('  Saving Page%d\'s %d img%s...' % (i, n - 1, 's' if n - 1 > 1 else ''))
    c = 0
    for j in range(n - 1):
      url = dataList[i][j].get('thumbURL')
      condition = save_img(url, path_save, timeout)
      if condition == 1:# success and not duplication
        c += 1
    print('  Saved Page%d\'s %d img%s.' % (i, c, 's' if c > 1 else ''))
  print()

if __name__ == '__main__':
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-k", "--keyword", required=True,
      help="the keyword for what you are searching")
  ap.add_argument("-o", "--output", required=True,
      help="path to saving the images")
  ap.add_argument("-p", "--pages", type=int, default=50,
      help="Number of pages")
  ap.add_argument("-t", "--timeout", type=float, default=3.05,
      help="Waiting time for each request (request time, connection time)")
  args = vars(ap.parse_args())

  save_images(keyword = args['keyword'], 
              path_save = args['output'], 
              pages = args['pages'], 
              timeout = args['timeout'])
  convert_images(args['output'])
