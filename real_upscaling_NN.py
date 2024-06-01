import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

# Name of the model used for upscaling.
# Different models will have different performance characteristics.
MODEL_NAME = 'RealESRGAN_x4plus'

# The factor by which the image is upscaled.
# This should match the scale the model was trained on.
SCALE = 4

# The size of the tiles the image is split into for processing.
# Smaller tiles reduce memory usage but might increase processing time.
TILE = 0

# The amount of padding added to each tile.
# More padding can help to reduce edge artifacts, but will also increase processing time.
TILE_PAD = 10

# Amount of padding added before splitting the image into tiles.
# This can help to reduce edge artifacts at tile borders.
PRE_PAD = 0

# Flag that determines whether to use half-precision floating point numbers for computation.
# Using half precision can speed up computation and reduce memory usage, but might slightly reduce the quality.
HALF = False

# Number of input/output channels.
# For color images, set this to 3. For grayscale images, set this to 1.
NUM_IN_CH = 3
NUM_OUT_CH = 3

# Number of feature maps in the RRDBNet architecture.
# Common values are powers of 2, like 64, 128, 256, etc.
# Higher values can improve results but increase computational requirements.
NUM_FEAT = 64

# Number of residual in residual dense blocks in the RRDBNet architecture.
# Common values are in the range of 10 to 30.
# Higher values can improve results but increase computational requirements.
NUM_BLOCK = 23

# Number of growth channels in the RRDBNet architecture.
# Common values are powers of 2, like 32, 64, 128, etc.
# Higher values can improve results but increase computational requirements.
NUM_GROW_CH = 32


def upscale_image(input_path, output_path, model_name=MODEL_NAME, scale=SCALE, tile=TILE, tile_pad=TILE_PAD, pre_pad=PRE_PAD, half=HALF):
    """
    Upscales an image using the Real-ESRGAN model.

    :param input_path: Path to the input image.
    :param output_path: Path to save the upscaled image.
    :param model_name: Name of the model to use for upscaling.
    :param scale: Factor by which to upscale the image.
    :param tile: Tile size for processing.
    :param tile_pad: Tile padding size.
    :param pre_pad: Pre-padding size.
    :param half: Whether to use half precision.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RRDBNet(num_in_ch=NUM_IN_CH, num_out_ch=NUM_OUT_CH, num_feat=NUM_FEAT, num_block=NUM_BLOCK, num_grow_ch=NUM_GROW_CH, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=f'weights/{model_name}.pth',
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        device=device
    )

    img = Image.open(input_path).convert('RGB')
    img_np = np.array(img)

    # Convert the image from RGB to YCbCr
    img_ycbcr = img.convert('YCbCr')
    img_np_ycbcr = np.array(img_ycbcr)

    with torch.no_grad():
        sr_img_np, _ = upsampler.enhance(img_np_ycbcr, outscale=scale)

    # Convert the image back to RGB
    sr_img_np = sr_img_np.round().astype(np.uint8)
    sr_img_ycbcr = Image.fromarray(sr_img_np, 'YCbCr')
    sr_img = sr_img_ycbcr.convert('RGB')

    sr_img.save(output_path)
    print(f"Upscaled image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = "shrek.png"
    output_image_path = "shrek_up.png"
    upscale_image(input_image_path, output_image_path)
