'''
Data: 2020-5-4
Auther: Yinghao Cai
Description: Extra images from local pdfs
Usage: python extra_imgs_from_pdfs.py --pdfs Pdfs
'''


# import necessary packages
import argparse
import filetype
import shutil
import fitz
import re
import os


def mkdirs_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_pdf(file_path):

  # judge if a directory
  if os.path.isdir(file_path):
    return False

  # judge if a pdf file
  kind = filetype.guess(file_path)
  return kind != None and kind.extension == 'pdf'


def pdf2pic(pdf_path, pic_base):
  checkXO = r"/Type(?= */XObject)"		# finds "/Type/XObject"
  checkIM = r"/Subtype(?= */Image)"		# finds "/Subtype/Image"

  doc = fitz.open(pdf_path)
  lenXREF = doc._getXrefLength()		# number of objects - do not use entry 0!

  for i in range(1, lenXREF):			# scan through all objects

    text = doc._getXrefString(i)          	# string defining the object
    isXObject = re.search(checkXO, text)    	# tests for XObject
    isImage = re.search(checkIM, text)      	# tests for Image
    if not isXObject or not isImage:        	# not an image object if not both True
      continue

    pix = fitz.Pixmap(doc, i)			# make pixmap from image
    pic_path = pic_base + '%d.png' % i  	# set png as default format

    if pix.n < 5:				# can be saved as PNG
      pix.writeImage(pic_path)
    else:					# must convert the CMYK first
      pix0 = fitz.Pixmap(fitz.csRGB, pix)
      pix0.writeImage(pic_path)
      pix0 = None				# free Pixmap resources
    pix = None		     			# free Pixmap resources


if __name__=='__main__':

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-p", "--pdfs", required=True,
                  help="path to all .pdf files")
  args = vars(ap.parse_args())

  # construct the structure of the path
  dirs = os.listdir(args["pdfs"])
  for filename in dirs:
    pdf_path = os.path.join(args["pdfs"], filename)

    # judge if a pdf file
    if not is_pdf(pdf_path):
      continue

    # construct pdf_dir and pic_dir
    pdf_dir = os.path.join(args["pdfs"], os.path.splitext(filename)[0])
    mkdirs_if_not_exists(pdf_dir)
    pic_dir = os.path.join(pdf_dir, "orig")
    mkdirs_if_not_exists(pic_dir)
    pic_dir1 = os.path.join(pdf_dir, "label")
    mkdirs_if_not_exists(pic_dir1)
    pic_dir2 = os.path.join(pdf_dir, "non-label")
    mkdirs_if_not_exists(pic_dir2)

    # construct pic_base
    pic_base = os.path.join(pic_dir, os.path.splitext(filename)[0])

    # move pdf_path to pic_dir/ and update pdf_path
    shutil.move(pdf_path, os.path.join(pdf_dir, filename))
    pdf_path = os.path.join(pdf_dir, filename)

    # extrate images from pdf_path
    try:
      pdf2pic(pdf_path, pic_base)
    except:
      pass

