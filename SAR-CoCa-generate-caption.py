import os
import csv
import open_clip
import torch
from PIL import Image
from tqdm import tqdm  # Progress bar

# Load model and preprocessing function
model, _, transform = open_clip.create_model_and_transforms(
    model_name="coca_ViT-L-14",
    pretrained="./checkpoints/epoch_3.pt"  # Path to the fine-tuned model
)

def generate_caption(image_path):
    """Generate a caption for a given image"""
    # Load and preprocess the image
    im = Image.open(image_path).convert("RGB")
    im = transform(im).unsqueeze(0)

    # Generate output using the model
    with torch.no_grad(), torch.amp.autocast("cuda"):
        generated = model.generate(im)

    # Decode and clean the generated text
    decoded_text = open_clip.decode(generated[0])
    cleaned_text = decoded_text.split("<end_of_text>")[0].replace("<start_of_text>", "").strip()
    return cleaned_text

def process_images_in_folder(folder_path, output_csv):
    """Batch process images in a folder and save captions to a CSV file"""
    # Get all image file paths (adjust extensions if needed)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Open CSV file and write header
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:  # Open file in append mode
        writer = csv.writer(file)
        
        # Write header if file is empty
        if os.stat(output_csv).st_size == 0:
            writer.writerow(['filepath', 'caption'])  # CSV file header

        # Loop through each image, generate a caption, and save to CSV
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(folder_path, image_file)
            
            try:
                # Check if the file has already been processed to avoid duplicates
                with open(output_csv, mode='r', encoding='utf-8') as check_file:
                    reader = csv.reader(check_file)
                    processed_files = [row[0] for row in reader if row]
                
                if image_path not in processed_files:
                    caption = generate_caption(image_path)
                    writer.writerow([image_path, caption])  # Write file path and caption
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue  # Continue processing remaining images

    # Count and print number of captions generated (excluding header)
    with open(output_csv, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        print(f"Total number of captions generated: {len(lines) - 1}")  # Subtract header line

# Example usage
folder_path = "./test_images"  # Set your image folder path
# Extract the last three parts of the path
path_parts = folder_path.strip(os.sep).split(os.sep)[-3:]  

# Create a new filename
output_csv = "SAR-CoCa-caption.csv"

process_images_in_folder(folder_path, output_csv)
