import json
import os

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


# List of your JSON files
BASE_PATH = "/opt/ml/aihub/label"
file_paths = [os.path.join(BASE_PATH, file_path) for file_path in os.listdir(BASE_PATH) if file_path.endswith('.json')]  # Ensure you're only listing .json files

# Load, transform, and combine JSON files
combined_json = load_and_transform_json(file_paths)

# To save the combined JSON to a file
with open('output.json', 'w', encoding='utf-8') as file:
    json.dump(combined_json, file, ensure_ascii=False, indent=4)

# File paths for the two input JSON files and the output file
file_path1 = '/opt/ml/level2-cv-datacentric-cv-13/target_data/ufo/train.json'
file_path2 = '/opt/ml/level2-cv-datacentric-cv-13/data/medical/ufo/train.json'
output_file_path = 'output.json'

# Merge the JSON files
merge_json_files(file_path1, file_path2, output_file_path)
