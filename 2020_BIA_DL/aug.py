import PIL
import numpy as np


def _gen_rect_coords_(center_x, center_y, length, width, angle):
  ar = np.pi * angle / 180
  sl2, sw2 = length/2, width/2
  cos, sin = np.cos(ar), np.sin(ar)
  slx2, sly2 = sl2 * cos, sl2 * sin
  swx2, swy2 = - sw2 * sin, sw2 * cos
  return [
          center_x + slx2 + swx2, center_y + sly2 + swy2,
          center_x - slx2 + swx2, center_y - sly2 + swy2,
          center_x - slx2 - swx2, center_y - sly2 - swy2,
          center_x + slx2 - swx2, center_y + sly2 - swy2
          ]

def _rand_in_range_(a, b):
  return a + (b - a) * np.random.rand()

def _gen_rand_rect_coords_(img_size, length_range, aspect_ratio_range, angle_range, x_range, y_range):
  iw, ih = img_size
  center = iw * _rand_in_range_(*x_range), ih * _rand_in_range_(*y_range)
  length = (iw + ih) * _rand_in_range_(*length_range)
  width = length * _rand_in_range_(*aspect_ratio_range)
  angle = _rand_in_range_(*angle_range)
  return _gen_rect_coords_(*center, length, width, angle)
  
def RandomRectFill(length_range=(0.01, 0.08), aspect_ratio_range=(0.05, 0.2), 
                   color=(0,0,0,255), angle_range=(0, 360), 
                   x_range=(0.1, 0.9), y_range=(0.3, 0.7), 
                   cnt_range=(1, 4), **kwargs):
  if isinstance(color, str):
    if color == 'blur':
      def func(img):
        mask = PIL.Image.new('1', img.size, color=0)
        drw = PIL.ImageDraw.Draw(mask)
        for i in range(np.random.randint(*cnt_range)):
          drw.polygon(_gen_rand_rect_coords_(img.size, length_range, aspect_ratio_range, angle_range, x_range, y_range), fill=1)
        flt = img.filter(PIL.ImageFilter.GaussianBlur(kwargs.get('blur_radius', 50)))
        img.paste(flt, (0, 0), mask)
        return img
      return func
    else:
      return lambda x: x
  else:
    def func(img):
      drw = PIL.ImageDraw.Draw(img)
      for i in range(np.random.randint(*cnt_range)):
        drw.polygon(_gen_rand_rect_coords_(img.size, length_range, aspect_ratio_range, angle_range, x_range, y_range), fill=color)
      return img
    return func
