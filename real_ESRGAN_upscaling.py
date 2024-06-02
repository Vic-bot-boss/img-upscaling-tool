import sys
import torch
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import numpy as np

# Name of the model used for upscaling.
MODEL_NAME = 'RealESRGAN_x4plus'
# Number of input/output channels.
NUM_IN_CH = 3
NUM_OUT_CH = 3
# Number of feature maps in the RRDBNet architecture.
NUM_FEAT = 64
# Number of residual dense blocks in the RRDBNet architecture.
NUM_BLOCK = 23
# Number of growth channels in the RRDBNet architecture.
NUM_GROW_CH = 32

def upscale_image(input_path, output_path, scale):
    """
    Upscales an image using the Real-ESRGAN model.

    :param input_path: Path to the input image.
    :param output_path: Path to save the upscaled image.
    :param scale: Factor by which to upscale the image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = RRDBNet(num_in_ch=NUM_IN_CH, num_out_ch=NUM_OUT_CH, num_feat=NUM_FEAT, num_block=NUM_BLOCK, num_grow_ch=NUM_GROW_CH, scale=scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=f'weights/{MODEL_NAME}.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
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
    if len(sys.argv) != 4:
        print("Usage: python real_upscaling_NN.py <input_path> <output_path> <scale>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    scale_factor = int(sys.argv[3])

    upscale_image(input_image_path, output_image_path, scale_factor)
