import os
from PIL import Image, ImageOps

def resize_and_convert_images(
    input_folder,
    output_folder,
    target_format="PDF",
    #target_size=(3840, 2160),
    bg_color=(255, 255, 255)  # White background for padding
):
    """
    Resizes images to 3840x2160 (maintaining aspect ratio) and converts to desired format.
    
    Args:
        input_folder (str): Folder containing images.
        output_folder (str): Where to save converted files.
        target_format (str): Output format (PDF, PNG, JPEG, etc.).
        target_size (tuple): Target resolution (width, height).
        bg_color (tuple): Background color for padding (R, G, B).
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Supported input formats
    supported_formats = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            try:
                # Open image
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)

                # Resize while maintaining aspect ratio (add padding if needed)
                #img = ImageOps.pad(img, target_size, color=bg_color, centering=(0.5, 0.5))

                # Save in target format
                output_name = os.path.splitext(filename)[0] + f".{target_format.lower()}"
                output_path = os.path.join(output_folder, output_name)

                if target_format.upper() == "PDF":
                    img.save(output_path, "PDF", resolution=100.0)
                else:
                    img.save(output_path, target_format.upper())

                #print(f"Processed: {filename} → {output_name} ({target_size[0]}x{target_size[1]})")
                print(f"Processed: {filename} → {output_name} ")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = r"C:\Users\User\Desktop\Image Conversion\Input Polder"
    output_folder = r"C:\Users\User\Desktop\Image Conversion\Output Folder"
    
    # Verify paths exist
    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        exit()
    
    target_format = "PDF"
    
    resize_and_convert_images(input_folder, output_folder, target_format)
    print("Conversion done!")