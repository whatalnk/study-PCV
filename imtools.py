import os
from PIL import Image
from numpy import *
def get_imlist(path):
  return [os.path.join(path, f) for f in os.listdir(path) if f.endwith('.jpg')]

def imresize(im, sz):
  pil_im = Image.fromarray(uint8(im))
  return array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
  imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
  cdf = imhist.cumsum()
  cdf = 255 * cdf / cdf[-1]
  im2 = interp(im.flatten(), bins[:-1], cdf)
  return im2.reshape(im.shape), cdf

def compute_average(imlist):
  averageim = array(Image.open(imlist[0]), 'f')
  for imname in imlist[1:]:
    try:
      averageim += array(Image.open(imname))
    except:
      print(imname + '...skipped')
  averageim /= len(imlist)
  return array(averageim, 'uint8')
