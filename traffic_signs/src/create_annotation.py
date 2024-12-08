import shutil
import json
import os
import pandas as pd
import yaml
from ultralytics import YOLO
from tqdm import tqdm

def get_df_annotations(annotations_file, label_map_file, min_area):
    with open(annotations_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    with open(label_map_file, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    id2label = {v: k for k, v in label_map.items()}
    df = pd.DataFrame(json_data['annotations'])
    df['sign_name'] = df['category_id'].map(id2label)
    df['global_group'] = df['sign_name'].apply(lambda x: x.split('_')[0])
    df = df[df['area'] >= min_area]

    return df

def get_balanced_df(df, samples_per_class):
    balanced_data = []
    for class_id in df['category_id'].unique():
        class_data = df[df['category_id'] == class_id]
        sample = class_data.sample(min(samples_per_class, len(class_data)),
                                   replace=False,
                                   random_state=42)
        balanced_data.append(sample)
    return pd.concat(balanced_data)

def get_filter_id(coco_json_train, label_map_file, samples_per_class, min_area=0, flag_filter = True):
    if flag_filter:    
        df_anno = get_df_annotations(coco_json_train, label_map_file, min_area)
        balanced_df = get_balanced_df(df_anno, samples_per_class)
        return balanced_df['id'].to_list()
    if not flag_filter:
        # Загрузка данных из COCO JSON
        with open(coco_json_train, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        with open(label_map_file, 'r', encoding='utf-8') as f:
            label_map = json.load(f)

        id2label = {v: k for k, v in label_map.items()}

        # Создание DataFrame из аннотаций
        df = pd.DataFrame(json_data['annotations'])
        df['sign_name'] = df['category_id'].map(id2label)
        
        # Удаление дубликатов на основе ID изображений и аннотаций
        df = df.drop_duplicates(subset=['image_id', 'category_id'])

        # Возвращаем список уникальных ID аннотаций
        return df['id'].to_list()

def convert_coco_to_yolo(coco_json, output_dir, image_dir, filter_anno=None, only_one_class=False):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    with open(coco_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    images_info = {image['id']: image for image in data['images']}

    for ann in tqdm(data['annotations']):
        if filter_anno and ann['id'] not in filter_anno:
            continue

        image_info = images_info.get(ann['image_id'])
        if not image_info:
            continue

        image_file_name = os.path.basename(image_info['file_name'])
        path_to_image = os.path.join(image_dir, image_file_name)

        if not os.path.exists(path_to_image):
            print(f"Image {image_file_name} not found.")
            continue

        category_id = 0 if only_one_class else ann['category_id'] - 1
        width, height = image_info['width'], image_info['height']
        x_center = (ann['bbox'][0] + ann['bbox'][2] / 2) / width
        y_center = (ann['bbox'][1] + ann['bbox'][3] / 2) / height
        bbox_width = ann['bbox'][2] / width
        bbox_height = ann['bbox'][3] / height

        yolo_format = f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

        label_file_name = os.path.splitext(image_file_name)[0] + '.txt'
        labels_output_path = os.path.join(output_dir, 'labels', label_file_name)
        images_output_path = os.path.join(output_dir, 'images', image_file_name)

        with open(labels_output_path, 'a') as file:
            file.write(yolo_format)

        shutil.copy(path_to_image, images_output_path)

# Конфигурация
config = {
    "root_dataset": r'C:\ДАТАСЕТ_КУРСАЧ\archive',
    "samples_per_class": 100,
    "min_area": 900,
    "only_one_class": False,
    "imgsz": 1280,
    "epochs": 35,
    "batch": 8,
    "output_dir": r'D:\STUDING_DENZA\4KYRS\ТИСПИС_курсач\working2\dataset'
}

coco_json_train = os.path.join(config["root_dataset"], 'train_anno.json')
coco_json_val = os.path.join(config["root_dataset"], 'val_anno.json')
label_map_file = os.path.join(config["root_dataset"], 'label_map.json')
path_to_images = os.path.join(config["root_dataset"], 'rtsd-frames')


flag_filter = True 
filter_annotation_id = get_filter_id(coco_json_train,
                                     label_map_file,
                                     config["samples_per_class"],
                                     config["min_area"],
                                     flag_filter = False)

convert_coco_to_yolo(coco_json_train,
                     os.path.join(config["output_dir"], 'train'),
                     path_to_images,
                     filter_annotation_id,
                     config["only_one_class"])

convert_coco_to_yolo(coco_json_val,
                     os.path.join(config["output_dir"], 'valid'),
                     path_to_images,
                     only_one_class=config["only_one_class"])

data_yaml = {
    'path': config["output_dir"],
    'train': 'train/images',
    'val': 'valid/images',
    'names': {0: 'traffic_sign'} if config["only_one_class"] else {}
}

labels_path = os.path.join(config["root_dataset"], 'labels.txt')
if not config["only_one_class"]:
    with open(labels_path, 'r') as file:
        class_names = [line.strip() for line in file]
        data_yaml['names'] = dict(enumerate(class_names))

with open(os.path.join(config["output_dir"], 'traffic_signs.yaml'), 'w') as file:
    yaml.dump(data_yaml, file)
