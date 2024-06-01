from PIL import Image

def upscale_image(input_path, output_path, scale_factor):
    """
    Upscales an image by a given scale factor.

    :param input_path: Path to the input image.
    :param output_path: Path to save the upscaled image.
    :param scale_factor: Factor by which to upscale the image.
    """
    # Open the input image
    with Image.open(input_path) as img:
        # Calculate the new dimensions
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)

        # Resize the image
        upscaled_img = img.resize((new_width, new_height), Image.LANCZOS)

        # Save the upscaled image
        upscaled_img.save(output_path)
        print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    input_image_path = "low-res-72dpi.jpg"
    output_image_path = "image.jpg"
    scale_factor = 2  # For example, to upscale by 2 times

    upscale_image(input_image_path, output_image_path, scale_factor)
