import json
import os
import numpy as np
import pandas as pd


class COCO2YOLO:
    def __init__(self, json_file, output ):
        self.json_file = json_file
        self.output = output
        self._check_file_and_dir(self.json_file, self.output)
        self.labels = json.load(open(self.json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\')+1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        x, y, width, height = bbox

        # Calculate YOLO format coordinates
        x_center = (x + width / 2) / img_w
        y_center = (y + height / 2) / img_h
        yolo_width = width / img_w
        yolo_height = height / img_h

        return x_center, y_center, yolo_width, yolo_height

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            arr = np.array([
                [obj[1]]+list(obj[2]) for obj in v
                ])
            pd.DataFrame(arr).to_csv(os.path.join(self.output, file_name), sep = " ", index = False, header = False)
            # print(arr)
            # import sys
            # sys.exit()
            # with open(os.path.join(self.output, file_name), 'w', encoding='utf-8') as f:
            #     print(len(v))
            #     print(v[0])
            #     # for obj in v:
            #     #     cat_name = self.coco_id_name_map.get(obj[1])
            #     #     category_id = self.coco_name_list.index(cat_name)
            #     #     box = ['{:.6f}'.format(x) for x in obj[2]]
            #     #     box = ' '.join(box)
            #     #     line = str(category_id) + ' ' + box
            #     #     f.write(line + '\n')

