from skimage.color.adapt_rgb import adapt_rgb, hsv_value
from skimage import exposure


@adapt_rgb(hsv_value)
def equalize_hist(image):
    return exposure.equalize_hist(image)
