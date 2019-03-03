import os
import os.path as osp
import json
import xml.etree.ElementTree as ET

"""
Prepare RGZ data into the COCO format based on
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

TODO - add host galaxy as a keypoint as done in the mask-rcnn paper
"""

IMG_SIZE = 132
NUM_CLASSES = 6 # or 6

CAT_XML_COCO_DICT = {'1C': 1, '2C': 2, '3C': 3, '1_1': 1, '1_2': 2, '1_3': 3,
                     '2_2': 4, '2_3': 5, '3_3': 6}

def create_categories():
    catlist = []
    if (3 == NUM_CLASSES):
        catlist.append({"supercategory": "galaxy", "id": 1, "name": "1C"})
        catlist.append({"supercategory": "galaxy", "id": 2, "name": "2C"})
        catlist.append({"supercategory": "galaxy", "id": 3, "name": "3C"})
    elif (6 == NUM_CLASSES):
        catlist.append({"supercategory": "galaxy", "id": 1, "name": "1C_1P"})
        catlist.append({"supercategory": "galaxy", "id": 2, "name": "1C_2P"})
        catlist.append({"supercategory": "galaxy", "id": 3, "name": "1C_3P"})
        catlist.append({"supercategory": "galaxy", "id": 4, "name": "2C_2P"})
        catlist.append({"supercategory": "galaxy", "id": 5, "name": "2C_3P"})
        catlist.append({"supercategory": "galaxy", "id": 6, "name": "3C_3P"})
    else:
        raise Exception('Incorrect NUM_CLASSES')
    return catlist

def create_coco_anno():
    anno = dict()
    anno['info'] = {"description": "RGZ data release 1", "year": 2018}
    anno['licenses'] = [{"url": r"http://creativecommons.org/licenses/by-nc-sa/2.0/", 
                         "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
    anno['images'] = []
    anno['annotations'] = []
    anno['categories'] = create_categories()
    return anno

def get_xml_metadata(img_id, xml_file, start_anno_id):
    ret = dict()
    tree = ET.parse(xml_file)
    ret['width'] = int(tree.find('size').find('width').text)
    ret['height'] = int(tree.find('size').find('height').text)
    objs = tree.findall('object')
    anno_list = []
    for idx, obj in enumerate(objs):
        anno = dict()
        anno['category_id'] = CAT_XML_COCO_DICT[obj.find('name').text]
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        bw = x2 - x1
        bh = y2 - y1
        anno['bbox'] = [x1, y1, bw, bh]
        anno['area'] = bh * bw #TODO mask will be different than this
        anno['id'] = start_anno_id + idx
        anno['image_id'] = img_id
        anno['iscrowd'] = 0
        anno_list.append(anno)
    ret['num_objs'] = len(objs)
    ret['anno_list'] = anno_list
    return ret

def xml2coco(img_list_file, in_img_dir, xml_dir, out_img_dir, json_dir):
    """
    convert the "old" rgz_rcnn format (xml) to claran coco format

    img_list_file:   a text file with a list of image names w/o file extensions (e.g. ".png")
    """
    anno = create_coco_anno()
    images = anno['images']
    with open(img_list_file, 'r') as fin:
        imglist = fin.read().splitlines()
    start_anno_id = 0
    for idx, img in enumerate(imglist):
        img_d = {'id': idx, 'license': 1, 'file_name': '%s.png' % img}
        xml_file = os.path.join(xml_dir, '%s.xml' % img)
        xml_meta = get_xml_metadata(idx, xml_file, start_anno_id)
        start_anno_id += xml_meta['num_objs']
        img_d['height'], img_d['width'] = xml_meta['height'], xml_meta['width']
        images.append(img_d)
        anno['annotations'].extend(xml_meta['anno_list'])
        if (idx % 300 == 0 and idx > 0):
            print("Processed %d xml files" % idx)
    json_dump = osp.join(json_dir, osp.splitext(osp.basename(img_list_file))[0] + '.json')
    with open(json_dump, 'w') as fout:
        json.dump(anno, fout)

if __name__ == '__main__':
    img_list_file = '/Users/chen/gitrepos/ml/' +\
                    'rgz_rcnn/data/RGZdevkit2017/RGZ2017/ImageSets/Main/testD4.txt'
    in_img_dir = None
    xml_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/Annotations'
    out_img_dir = None
    json_dir = '.'
    xml2coco(img_list_file, in_img_dir, xml_dir, out_img_dir, json_dir)


        
