import json
import os
import shutil
from random import sample


def transform_json(old_json):
    new_json = {"images": {}}

    for image in old_json["images"]:
        file_name = image["image.file.name"]
        img_width = image["image.width"]
        img_height = image["image.height"]
        
        new_image = {
            file_name: {
                "paragraphs": {},
                "words": {},
                "chars": {},
                "img_w": img_width,
                "img_h": img_height,
                "tags": ["autoannotated"],
                "relations": {},
                "annotation_log": {
                    "worker": "worker",
                    "timestamp": "2023-01-26",
                    "tool_version": "",
                    "source": None
                },
                "license_tag": {
                    "usability": True,
                    "public": False,
                    "commercial": True,
                    "type": None,
                    "holder": "AIHub"
                }
            }
        }

        for annotation in old_json["annotations"]:
            x, y, w, h = annotation["annotation.bbox"]
            word_id = "some_unique_id"  # Generate or use a unique ID for each word
            new_image[file_name]["words"][word_id] = {
                "transcription": annotation["annotation.text"],
                "points": [[x, y], [x + w, y], [x, y + h], [x + w, y + h]],
                "orientation": "Horizontal",
                "language": None,
                "tags": ["Auto"],
                "confidence": None,
                "illegibility": False
            }

        new_json["images"].update(new_image)

    return new_json


def load_and_transform_json(file_paths):
    all_transformed_data = {}  # Initialize as a dictionary, not a list
    
    for file_path in file_paths:
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            old_json = json.load(file)

        # Transform the JSON data
        transformed_json = transform_json(old_json)
        
        # Merge the transformed data into the main dictionary
        all_transformed_data.update(transformed_json["images"])

    return {"images": all_transformed_data}


def merge_json_files(file_path1, file_path2, output_file_path):
    # Load the first JSON file
    with open(file_path1, 'r', encoding='utf-8') as file:
        data1 = json.load(file)
    
    # Load the second JSON file
    with open(file_path2, 'r', encoding='utf-8') as file:
        data2 = json.load(file)
    
    # Merge the 'images' dictionaries from both JSON files
    merged_images = {**data1['images'], **data2['images']}
    
    # Create the new combined JSON structure
    combined_json = {'images': merged_images}

    # Save the combined JSON to a file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(combined_json, file, ensure_ascii=False, indent=4)


# Directory and file paths
AIHUB_PATH = '/data/ephemeral/home/aihub'
RAW_DATA_PATH = '/data/ephemeral/home/data/medical'
OUTPUT_PATH = '/data/ephemeral/home/data/aihub670'

IMG_DIR = f"{AIHUB_PATH}/img"
LABEL_DIR = f"{AIHUB_PATH}/label"

RAW_LABEL_PATH = f'{RAW_DATA_PATH}/ufo/train.json'
RAW_IMG_TRAIN_PATH = f'{RAW_DATA_PATH}/img/train'
RAW_IMG_TEST_PATH = f'{RAW_DATA_PATH}/img/test'

OUTPUT_IMG_TRAIN_DIR = f"{OUTPUT_PATH}/img/train"
OUTPUT_IMG_TEST_DIR = f"{OUTPUT_PATH}/img/test"
OUTPUT_UFO_DIR = f"{OUTPUT_PATH}/ufo"

num_files = 670  # Set the number of files you want to process

# Ensure output directories exist
os.makedirs(OUTPUT_IMG_TRAIN_DIR, exist_ok=True)
os.makedirs(OUTPUT_IMG_TEST_DIR, exist_ok=True)
os.makedirs(OUTPUT_UFO_DIR, exist_ok=True)

# Copy all test images from RAW_IMG_TEST_PATH to OUTPUT_IMG_TEST_DIR
for img_file in os.listdir(RAW_IMG_TEST_PATH):
    src_file = os.path.join(RAW_IMG_TEST_PATH, img_file)
    shutil.copy(src_file, OUTPUT_IMG_TEST_DIR)

# Get all file names (without extension) from LABEL_DIR
file_names = [file_name.replace('.json', '') for file_name in os.listdir(LABEL_DIR) if file_name.endswith('.json')]

# Select a subset of file names if num_files is set
if num_files is not None and num_files < len(file_names):
    file_names = sample(file_names, num_files)

# Full paths for label and image files
label_files = [os.path.join(LABEL_DIR, f"{file_name}.json") for file_name in file_names]
img_files = [os.path.join(IMG_DIR, f"{file_name}.jpg") for file_name in file_names]

# Copy the sampled images from AIHUB_PATH to OUTPUT_IMG_TRAIN_DIR
for img_file in img_files:
    shutil.copy(img_file, OUTPUT_IMG_TRAIN_DIR)

# Copy all train images from RAW_IMG_TRAIN_PATH to OUTPUT_IMG_TRAIN_DIR
for img_file in os.listdir(RAW_IMG_TRAIN_PATH):
    src_file = os.path.join(RAW_IMG_TRAIN_PATH, img_file)
    shutil.copy(src_file, OUTPUT_IMG_TRAIN_DIR)

# Load, transform, and combine JSON files
combined_json = load_and_transform_json(label_files)

# Save the combined JSON to a temporary file
temp_json_path = os.path.join(OUTPUT_UFO_DIR, 'temp.json')
with open(temp_json_path, 'w', encoding='utf-8') as file:
    json.dump(combined_json, file, ensure_ascii=False, indent=4)

# Output file path for the merged JSON
output_file_path = os.path.join(OUTPUT_UFO_DIR, 'train.json')

# Merge the JSON files
merge_json_files(temp_json_path, RAW_LABEL_PATH, output_file_path)

# Optionally, remove the temporary JSON file
os.remove(temp_json_path)
