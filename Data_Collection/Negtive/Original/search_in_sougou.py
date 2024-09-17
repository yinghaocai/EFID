# -*- coding: UTF-8 -*-
# usage0:
# python search_in_sougou.py --keyword 岩石 --output 'sougou_rock_images' --pages 50 --timeout 3.05
# usage1:
# python search_in_sougou.py --keyword 石头 --output 'sougou_stone_images' --pages 50 --timeout 3.05

# import necessary packages
from common import mkdirs_if_not_exists, legitimize, save_img, convert_images
from bs4 import BeautifulSoup
import argparse
import urllib
import os
import re

def save_images(keyword, path_save, pages, timeout = 2):
  # default images per page
  images = 48

  mkdirs_if_not_exists(path_save)
  web_basic = 'https://pic.sogou.com/pics?query={0}&mode=1&start={1}&reqType=ajax&reqFrom=result&tn=0&len={2}'
  
  # generate links0 of page's web
  print('  Getting %d Page\' url%s...' % (pages, 's' if pages > 1 else ''))
  webs=[]
  for i in range(pages):
    web = legitimize(web_basic.format(keyword, i * images, images))
    webs.append(web)
  print('  Got %d Page\' url%s.\n' % (len(webs), 's' if len(webs) > 1 else ''))
  
  for i in range(pages):
    web = urllib.request.Request(webs[i])
    web=urllib.request.urlopen(web)
    soup = BeautifulSoup(web.read(), 'html.parser')
    urls = re.findall(r"\"pic_url\":\"(.+?)\"", str(soup))
    
    print('  Saving Page%d\'s %d img%s...' % (i, len(urls), 's' if len(urls) > 1 else ''))
    c = 0
    for url in urls:
      url = legitimize(url)
      condition = save_img(url, path_save, timeout = timeout)
      if condition == 1:
        c += 1
    print('  Saved Page%d\'s %d image%s.' % (i, c, 's' if c > 1 else ''))
    web.close()
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

  save_images(args['keyword'], 
              path_save = args['output'], 
              pages = args['pages'], 
              timeout = args['timeout'])
  convert_images(args['output'])
