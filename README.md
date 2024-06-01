I often find myself using online tools for resizing and upscaling with feature hidden behind paywalls. So here's mine...

# Image Upscaling Tool

This repository contains tools for upscaling images using Real-ESRGAN and Pillow.

## Tools

### 1. Real-ESRGAN Upscaling

#### Script: `real_upscaling_NN.py`

Upscales images using the Real-ESRGAN model.

**Usage:**
```bash
python real_upscaling_NN.py
```

**Parameters:**
- `MODEL_NAME`: Name of the model used for upscaling.
- `SCALE`: Factor by which the image is upscaled.
- `TILE`: Size of the tiles the image is split into for processing.
- `TILE_PAD`: Amount of padding added to each tile.
- `PRE_PAD`: Amount of padding added before splitting the image into tiles.
- `HALF`: Whether to use half-precision floating point numbers for computation.
- `NUM_IN_CH`: Number of input channels.
- `NUM_OUT_CH`: Number of output channels.
- `NUM_FEAT`: Number of feature maps in the RRDBNet architecture.
- `NUM_BLOCK`: Number of residual dense blocks in the RRDBNet architecture.
- `NUM_GROW_CH`: Number of growth channels in the RRDBNet architecture.

### 2. Pillow Resizing

#### Script: `pillow_resizing.py`

Upscales images using the Pillow library. Only changes resolution!

**Usage:**
```bash
python pillow_resizing.py
```

**Parameters:**
- `input_image_path`: Path to the input image.
- `output_image_path`: Path to save the upscaled image.
- `scale_factor`: Factor by which to upscale the image.

## Example Results

### Before and After

#### Example 1:
**Before:** ![low-res-72dpi](low-res-72dpi.jpg)
**After (Real-ESRGAN):** ![image](image.jpg)

#### Example 2:
**Before:** ![shrek](shrek.png)
**After (Real-ESRGAN):** ![shrek_up](shrek_up.png)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/img-upscaling-tool.git
cd img-upscaling-tool
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Download the weights and place them in the `weights/` directory:
```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
```

