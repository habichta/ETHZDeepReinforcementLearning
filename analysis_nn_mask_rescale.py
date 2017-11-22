
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_constants import c_img_path
import skimage.io as skio
import skimage.transform as skt
import os



mask_path = os.path.join(c_img_path,"cavriglia_skymask.png")
mask_output_path = os.path.join(c_img_path,"cavriglia_skymask256.png")

image = skio.imread(mask_path)

print(image.shape)

image = skt.resize(image, (256, 256))

print(image.shape)

skio.imsave(mask_output_path, image)