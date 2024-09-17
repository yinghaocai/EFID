'''
Data: 2020-5-3
Auther: Yinghao Cai
Description: Download all pdfs from links_file created by running extra_from_web.py
usage python extra_links_from_web.py --pdfs pdf_links --invalid invalid_links --url http://www.xinglida.net
'''

# import necessary packages
from bs4 import BeautifulSoup
import urllib.request
import threading
import argparse
import requests
import os

def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def is_url_exits(url, timeout = 2):
  try:
    r = requests.get(url, timeout=timeout)
    code = r.status_code
    return code == 200
  except:
    print(url)
    return False

# find out all links in pages
def get_url(base_url):

  links = []
  if not is_url_exits(base_url, 2):
    return links
  try:
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, 'html.parser')
  except:
    return links

  for a in soup.find_all('a'):
    try:
      # Had not considered './' or '../' in url yet
      if a['href'].find("://") == -1:
        link = os.path.join(base_url, a['href'])
      else:
        link = a['href']
      links.append(link)
    except:
      pass
  return links


if __name__ == "__main__":

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-u", "--url", required=True,
                  help="url to the web.")
  ap.add_argument("-p", "--pdfs", required=True,
                  help="path to save the pdf format links.")
  ap.add_argument("-i", "--invalid", required=True,
                  help="path to save the other format links.")
  args = vars(ap.parse_args())

  # open the files to save the links
  fp = open(args['pdfs'], 'w')
  fi = open(args['invalid'], 'w')

  # save link to Corresponding file
  links = get_url(args["url"])
  for link in links:

    # simple judgement about pdf is enough, 
    # because further judgement will happen in save_pdf_from_link.py with filetype package.
    if link.endswith(".pdf") or link.endswith(".PDF"):
      print(link, file = fp)
    else:
      print(link, file = fi)

  # close the files
  fp.close()
  fi.close()
