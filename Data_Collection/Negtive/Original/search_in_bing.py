# -*- coding: UTF-8 -*-
# usage0:
# python search_in_bing.py --keyword 岩石 --output 'bing_rock_images' --pages 23 --timeout 3.05
# usage1:
# python search_in_bing.py --keyword 石头 --output 'bing_stone_images' --pages 21 --timeout 3.05

# import necessary packages
from common import mkdirs_if_not_exists, legitimize, save_img, convert_images
from bs4 import BeautifulSoup
import argparse
import urllib
import os

  
# save imgs from dynamic web
def save_images(keyword, path_save, pages, timeout):
  # default images per page
  images = 35

  
  # avoid repeat request
  def find_set(s, v):
    m = len(s)
    s.discard(v)
    n = len(s)
    if m == n + 1:
      s.add(v)
      return True
    return False


  # extra the images'url to a list
  def extra_urls(url):
    url = legitimize(url)
    web = urllib.request.Request(url)
    web = urllib.request.urlopen(web)
    soup = BeautifulSoup(web.read(), 'html.parser')
    links = []
    for StepOne in soup.find_all(name = 'a'):
      if str(StepOne).find('murl') != -1:
        try:
          m = StepOne.attrs['m']
          links.append(eval(m)['murl'])
        except:
          pass
    return links

  # construct basic params
  url = 'https://cn.bing.com/images/async?q={0}&first={1}&count={2}&repl={3}&lostate=r&mmasync=1&dgState=x*175_y*848_h*199_c*1_i*106_r*0'
  keyword = legitimize(keyword)
  mkdirs_if_not_exists(path_save)
  Set = set({})
  
  # generate links0 of page's web
  print('  Getting %d Page\' url%s...' % (pages, 's' if pages > 1 else ''))
  webs = []
  for i in range(pages):
    web = url.format(keyword, i * images, images, images)
    try:
      urllib.request.Request(web)
      webs.append(web)
    except:
      break
  print('  Got %d Page\' url%s.\n' % (len(webs), 's' if len(webs) > 1 else ''))
  
  # find out images's url and save them respectly
  for i in range(len(webs)):
    c = 0
    urls = extra_urls(webs[i])
    print('  Saving Page%d\'s %d img%s...' % (i, len(urls), 's' if len(urls) > 1 else ''))
    for url in urls:
      if find_set(Set, url):
        continue
      condition = save_img(url, path_save, timeout)
      if condition == 1:# success and not duplication
        c += 1
      elif condition == 2:# download error
        Set.add(url)
    print('  Saved Page%d\'s %d img%s.' % (i, c, 's' if c > 1 else ''))
  print()
  
  # Pick up the leak
  print('  Picking up the leak %d image%s in the final...' % (len(Set), 's' if len(Set) > 1 else ''))
  c = 0
  timeout = (3.04, 27)
  for s in Set:
    condition = save_img(s, path_save, timeout = timeout)
    if condition == 1:
      c += 1
  print('  Picked up %d image%s.' % (c, 's' if c > 1 else ''))
  print()
  
if __name__=='__main__':
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-k", "--keyword", required=True,
      help="the keyword for what you are searching")
  ap.add_argument("-o", "--output", required=True,
      help="path to saving the images")
  ap.add_argument("-p", "--pages", type=int, default=20,
      help="Number of pages")
  ap.add_argument("-t", "--timeout", type=float, default=3.04,
      help="Waiting time for each request (request time, connection time)")
  args = vars(ap.parse_args())
  
  save_images(keyword = args['keyword'], path_save = args['output'], pages = args['pages'], timeout = args['timeout'])
  convert_images(path = args['output'])
