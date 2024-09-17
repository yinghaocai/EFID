'''
Data: 2020-5-5
Auther: Yinghao Cai
Description: Download all pdfs from links_file created by running extra_from_web.py
usage python save_pdf_from_link.py --pdfs Pdfs --links pdf_links
'''


# import necessary packages
import argparse
import filetype
import requests
import os


def mkdir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_pdf(file_path):

  # download file failed or it's a directory file
  if not os.path.exists(file_path) or os.path.isdir(file_path):
    return False

  # judge the file if a pdf file
  kind = filetype.guess(file_path)
  if kind == None or kind.extension != 'pdf':
    os.remove(file_path)
    return False
  return True


def save_pdf(src, target):
  try:
    src_file = requests.get(src, stream = False)
    with open(target,"wb") as target_file:
      target_file.write(src_file.content)
  except:
    print('Failed to download pdf from %s' % src)


if __name__ == "__main__":

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--pdfs", required=True,
                  help="path to save the pdf's directory.")
  ap.add_argument("-l", "--links", required=True,
                  help="path to the links file.")
  args = vars(ap.parse_args())

  # initial basic parameters
  pdfs_dir = args['pdfs']
  mkdir_if_not_exists(pdfs_dir)
  links_path = args['links']

  # count unsuccessfully and successfully downloading pdf
  unsuc = 0
  suc = 0

  # save pdf from link
  with open(links_path, 'r') as links_file:
    for line in links_file:
      src = line.replace('\n', '').replace('%20', ' ')
      target = os.path.join(pdfs_dir, os.path.split(src)[-1]).replace('%20', ' ')
      save_pdf(src, target)
      if is_pdf(target):
        suc += 1
      else:
        unsuc += 1
  print('%d pdf file(s) download successfully.' % suc)
  print('%d pdf file(s) download failed.' % unsuc)

