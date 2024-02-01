import os
import shutil

source_dir = '/data/ephemeral/home/aihub-raw'

# 대상 디렉토리 경로
destination_img_dir = '/data/ephemeral/home/aihub/img'
destination_label_dir = '/data/ephemeral/home/aihub/label'

# 대상 디렉토리 생성
os.makedirs(destination_img_dir, exist_ok=True)
os.makedirs(destination_label_dir, exist_ok=True)

# 원본 디렉토리 순회
for root, _, files in os.walk(source_dir):
    for file in files:
        source_file_path = os.path.join(root, file)
        if source_file_path.endswith('.json'):
            destination_file_path = os.path.join(destination_label_dir, file)
            shutil.copy(source_file_path, destination_file_path)
        elif file.endswith('.jpg'):
            destination_file_path = os.path.join(destination_img_dir, file)
            shutil.copy(source_file_path, destination_file_path)

# Define the directories
images_dir = destination_img_dir
labels_dir = destination_label_dir

# Get the list of image and label files
image_files = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]
label_files = [f.split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.json')]

# Find files in images that are not in labels and vice versa
images_without_labels = [f"{img}.jpg" for img in image_files if img not in label_files]
labels_without_images = [f"{lbl}.json" for lbl in label_files if lbl not in image_files]

# Delete the image files without corresponding labels
for img in images_without_labels:
    os.remove(os.path.join(images_dir, img))

# Delete the label files without corresponding images
for lbl in labels_without_images:
    os.remove(os.path.join(labels_dir, lbl))

print("Construction complete.")
