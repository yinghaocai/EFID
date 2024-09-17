# import necessary packages
from filetype import guess
from PIL import Image
import requests
import urllib
import os


def mkdirs_if_not_exists(path):
  if not os.path.exists(path):
    os.makedirs(path)

# process Chinese character
def legitimize(link):
  s = ''
  for c in link:
    if c > chr(127):
      s = s + urllib.parse.quote(c)
    else:
      s = s + c
  return s
  

# save the file whatever its type is
def save_img(url, path, timeout):
  url = legitimize(url)
  file_name = url.split('/')[-1]
  short_name = ''
  if len(file_name) > 110:
    short_name = file_name[:100] + file_name[len(file_name)-10:]
  else:
    short_name = file_name
  file_path = os.path.join(path, short_name)
  if os.path.exists(file_path):
    return 0
  try:
    file_url = requests.get(url, timeout = timeout)
    with open(file_path,"wb") as fp:
      fp.write(file_url.content)
    return 1
  except:
    # print('  Failed to download img from %s\n' % url)
    return 2


# convert to png format to overwrite itself
def convert_images(path):
  imgs = [os.path.join(path, img) for img in os.listdir(path)]
  print('  Converting %s\'s %d image%s...' % (path, len(imgs), 's' if len(imgs) > 1 else ''))
  c = 0
  for img in imgs:
    if os.path.isdir(img):
      continue

    kind = guess(img)
    if kind == None:
      os.remove(img)
      continue
    try:
      # convert
      with Image.open(img) as fp:
        fp.save(img, format = 'png')
      # rename
      png = os.path.splitext(img)[0] + '.png'
      os.rename(img, png)
      c += 1
    except:
      os.remove(img)
  imgs = [os.path.join(path, img) for img in os.listdir(path)]
  print('  Converted %s\'s %d image%s.\n' % (path, c, 's' if c > 1 else ''))

