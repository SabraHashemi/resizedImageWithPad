import cv2
import numpy as np
import cv2

from typing import Tuple

def resize_with_pad(image: np.array, new_shape: Tuple[int, int], padding_color: Tuple[int] = (0, 0, 0)) -> np.array:
      original_shape = (image.shape[1], image.shape[0])
      ratio = float(max(new_shape))/max(original_shape)
      new_size = tuple([int(x*ratio) for x in original_shape])
  
      if new_size[0] > new_shape[0] or new_size[1] > new_shape[1]:
          ratio = float(min(new_shape)) / min(original_shape)
          new_size = tuple([int(x * ratio) for x in original_shape])
  
      image = cv2.resize(image, new_size)
      ##delta_w = new_shape[0] - new_size[0]
      ##delta_h = new_shape[1] - new_size[1]
      delta_w = new_shape[0] - new_size[0] if new_shape[0] > new_size[0] else 0
      delta_h = new_shape[1] - new_size[1] if new_shape[1] > new_size[1] else 0
      top, bottom = delta_h//2, delta_h-(delta_h//2)
      left, right = delta_w//2, delta_w-(delta_w//2)
  
      image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,None,value=padding_color)
      return image


def resizeAndPad2(img, size, padColor=255):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA

    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h 
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img


def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size
    orig_aspect = w/h
    desired_aspect = sw/sh

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h

        

    # compute scaling and pad sizing
    if w < sw: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif  h < sh: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img






# Saving the new image inside DATA folder
path = '/home/sabra-pc/Datasets/car1/000001.jpg'
img = cv2.imread(path)
new_img = resizeAndPad2(img, (320,240), 0)
  
# Using cv2.imwrite() method
# Saving the image
cv2.imshow("h", new_img)
cv2.waitKey(0)

image = resize_with_pad(img, (320,240))
cv2.imshow("Padded image", image)
cv2.waitKey()
