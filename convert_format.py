import json


def convert_ufo_to_coco_format(input_path: str, output_path: str):
    new_coco = {}
    new_images = []
    new_annotations = []
    new_categories = [
        {
            "id": 1,
            "name": "stamp",
        },
        {
            'id': 2,
            'name': 'Auto'
        }
                      
    ]
    cat_to_idx = {
        "stamp": 1,
        'Auto': 2
    }
    
    with open(input_path, 'r') as f:
        ufo_json = json.load(f)
    
    images = ufo_json['images']
    annot_id = 1
    # images
    for img_id, (file_name, data) in enumerate(images.items()):
        new_image = {
            'id': img_id+1,
            'file_name': file_name,
            'height': data['img_h'],
            'width': data['img_w']
        }
        new_images.append(new_image)
        
        for _, annot in data['words'].items():
            pts = annot['points']
            
            min_x, min_y = min(pts, key=lambda x: x[0])[0], min(pts, key=lambda x: x[1])[1]
            max_x, max_y = max(pts, key=lambda x: x[0])[0], max(pts, key=lambda x: x[1])[1]
            w, h = max_x - min_x, max_y - min_y
            
            ## category_id
            cat_id = -1
            category = annot['tags']
            mask_flag = any(map(lambda x: x in ['masked', 'maintable', 'stamp'], category))
            
            ## must be masking
            category = 'stamp' if mask_flag else 'Auto'            
            cat_id = cat_to_idx[category]
            
            ## annotations
            new_annot = {
                "id": annot_id,
                "image_id": img_id+1,
                "category_id": cat_id,
                "segmentation": [[value for sublist in pts for value in sublist]],
                "area": w * h,
                "bbox": [min_x, min_y, w, h],
                "iscrowd": 0
            }
            new_annotations.append(new_annot)
            annot_id += 1
    
    new_coco['images'] = new_images
    new_coco['annotations'] = new_annotations
    new_coco['categories'] = new_categories
    
    with open(output_path, 'w') as f:
        json.dump(new_coco, f, indent=4)    


def convert_coco_to_ufo_format(coco_input_path: str, coco_output_path: str):
    ufo_format = {
        "images": {}
    }
    
    with open(coco_input_path, 'r') as f:
        coco_json = json.load(f)
        
    images = coco_json['images']
    annotations = coco_json['annotations']
    categories = {k['id']: k['name'] for k in coco_json['categories']}
    
    for image in images:
        img_id = image['id']
        file_name = image['file_name']
        img_h = image['height']
        img_w = image['width']
        
        ufo_format['images'][file_name] = {
            "paragraphs": {},
            "words": {},
            "chars": {},
            'img_h': img_h,
            'img_w': img_w,
            'words': {},
            "relations": {},
            "tags": ["re-labeled"]
        }
        
        annot_id = 1
        candits = [annot for annot in annotations if annot['image_id'] == img_id]
        for candit in candits:
            try:
                points = [candit['segmentation'][0][i:i+2] for i in range(0, len(candit['segmentation'][0]), 2)]
            except:
                x1, y1, w, h = candit['bbox']
                x2, y2 = x1 + w, y1 + h
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            tag = categories[candit['category_id']]
            ufo_format['images'][file_name]['words'][str(annot_id).zfill(4)] = {
                "transcription": "",
                'points': points,
                "orientation": "Horizontal",
                'tags': [tag],
                "illegibility": False,
                "language": None,
                "confidence": None
            }
            annot_id += 1

    with open(coco_output_path, "w") as f:
        json.dump(ufo_format, f, indent=4)


if __name__ == '__main__':
    input_path = './data/medical/ufo/train.json'
    output_path = './data/medical/ufo/train_coco.json'
    convert_ufo_to_coco_format(input_path, output_path)
    
    coco_input_path = './data/medical/ufo/relabel_train_coco.json'
    coco_output_path = './data/medical/ufo/relabel_train.json'
    convert_coco_to_ufo_format(coco_input_path, coco_output_path)
    