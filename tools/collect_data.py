import os
import dicttoxml
import numpy
import numpy as np
import random
import lxml.etree
import cv2
from skimage.util import random_noise

path_data = '/home/sergei/DATA_HS'

jpeg_path = '/home/sergei/DATA_HS/jpeg'
milk_200_hard_path = '/home/sergei/DATA_HS/milk_200_hard'
milk365_path = '/home/sergei/DATA_HS/milk365'
remainder_path = '/home/sergei/DATA_HS/remainder'

save_path_images_train = '/home/sergei/HS_TRAIN/JPEGImages'
save_path_annotetions_train = '/home/sergei/HS_TRAIN/Annotations'

save_path_images_val = '/home/sergei/HS_VAL/JPEGImages'
save_path_annotetions_val = '/home/sergei/HS_VAL/Annotations'

scale = 0.5
random.seed(10)

def splitDir(path: str, split_p=0.8, extension='.tiff'):
    path_img = path + '/JPEGImages'
    path_xml = path + '/Annotations'

    image = os.listdir(path_img)
    name = [os.path.splitext(name)[0] for name in image]
    random.shuffle(name)
    idx = int(len(name)*split_p)
    train_name = name[0: idx]
    val_name = name[idx:]

    train_image = [path_img + '/' + img + extension for img in train_name]
    train_xml = [path_xml + '/' + name + '.xml' for name in train_name]

    val_image = [path_img + '/' + img + extension for img in val_name]
    val_xml = [path_xml + '/' + name + '.xml' for name in val_name]

    return train_image, train_xml, val_image, val_xml

def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def AddObjectToAnnotete(annotete: dict, xmin: int, ymin: int, xmax:int, ymax:int, clear = False):

    if clear:
        annotete['annotation']['object'].clear()

    else:
        obj = {'name': 'honest_sign', 'truncated': '0', 'occluded': '0', 'difficult': '0',
               'bndbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
               'attributes': {'attribute': {'name': 'rotation', 'value': '0'}}}

        annotete['annotation']['object'].append(obj)

def SetSize(annotete, width: int, height: int):
    annotete['annotation']['size']['width'] = width
    annotete['annotation']['size']['height'] = height

def RoteteBox(xmin, ymin, xmax, ymax, angel=0):

    ox = (xmin+xmax)/2.0
    oy = (ymax+ymin)/2.0



    angel = np.deg2rad(angel)

    x1 = ox + (xmin - ox)*np.cos(angel) + (ymin-oy)*np.sin(angel)
    y1 = oy + (xmin-ox)*np.sin(angel) + (ymin-oy)*np.cos(angel)

    x2 = ox + (xmax - ox)*np.cos(angel) + (ymax-oy)*np.sin(angel)
    y2 = oy + (xmax-ox)*np.sin(angel) + (ymax-oy)*np.cos(angel)

    x3 = xmax
    y3 = ymin

    x4 = xmin
    y4 = ymax

    x3 = ox + (x3 - ox) * np.cos(angel) +(y3 - oy) * np.sin(angel)
    y3 = oy + (x3 - ox) * np.sin(angel) + (y3 - oy) * np.cos(angel)

    x4 = ox + (x4 - ox) * np.cos(angel) + (y4 - oy) * np.sin(angel)
    y4 = oy + (x4 - ox) * np.sin(angel) + (y4 - oy) * np.cos(angel)

    sx = np.sort([x1,x2,x3,x4])
    sy = np.sort([y1,y2,y3,y4])

    xmin = sx[0]
    ymin = sy[0]
    xmax = sx[3]
    ymax = sy[3]

    return int(xmin), int(ymin), int(xmax), int(ymax)

def SaveSample(path_image, path_xml, image, xml, prefix = ''):
    save_name = os.path.splitext(xml['annotation']['filename'])[0]
    xml['annotation']['filename'] = save_name + '_' + prefix + '.jpg'
    cv2.imwrite(os.path.join(path_image, save_name + '_' + prefix + '.jpg'), image)
    save_xml = dicttoxml.dicttoxml(xml, root=False, attr_type=False, cdata=False)
    myfile = open(os.path.join(path_xml, save_name+ '_' + prefix + '.xml'), "wb")
    myfile.write(save_xml)
    myfile.close()

cv2.namedWindow('TT', cv2.WINDOW_NORMAL)
def GenSample(path_image, path_annotate, scale):
    annotation_xml = lxml.etree.fromstring(open(path_annotate).read())
    annotation_xml = parse_xml(annotation_xml)


    width = int(int(annotation_xml['annotation']['size']['width'])*scale)
    height = int(int(annotation_xml['annotation']['size']['height'])*scale)

    scale_bbox_w = height/width

    #print(scale_bbox_w)

    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (height, height))
    #print(width, height)
    #print(image.shape)

    try:
        objects = annotation_xml['annotation']['object'].copy()
        AddObjectToAnnotete(annotation_xml, 0, 0, 0, 0, True)
        for obj in objects:
            xmin = float(obj['bndbox']['xmin'])*scale_bbox_w
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])*scale_bbox_w
            ymax = float(obj['bndbox']['ymax'])
            xmin, ymin, xmax, ymax = RoteteBox(xmin, ymin, xmax, ymax, float(obj['attributes']['attribute']['value']))
            #print('angel: {}'.format(float(obj['attributes']['attribute']['value'])))

            xmin = int(xmin * scale)
            ymin = int(ymin * scale)
            xmax = int(xmax * scale)
            ymax = int(ymax * scale)
            AddObjectToAnnotete(annotation_xml, xmin, ymin, xmax, ymax)

        SetSize(annotation_xml, height, height)
        SaveSample(save_path_images_train, save_path_annotetions_train, image, annotation_xml, '1')
        '''cv2.rectangle(image, (xmin,ymin), (xmax, ymax), 255, thickness=1)
        cv2.imshow('TT', image)
        cv2.waitKey()'''

        # Add salt-and-pepper noise to the image.
        noise_img = random_noise(image, mode='s&p', amount=0.001)
        noise_img = np.array(255 * noise_img, dtype='uint8')
        SaveSample(save_path_images_train, save_path_annotetions_train, noise_img, annotation_xml, '2')

        noise_img = random_noise(image, mode='gaussian', mean=0, var=0.0015)
        noise_img = np.array(255 * noise_img, dtype='uint8')
        SaveSample(save_path_images_train, save_path_annotetions_train, noise_img, annotation_xml, '3')

        image = np.array(image * 1.2, dtype='uint8')
        SaveSample(save_path_images_train, save_path_annotetions_train, image, annotation_xml, '4')

        image = np.array(image * 0.7, dtype='uint8')
        SaveSample(save_path_images_train, save_path_annotetions_train, image, annotation_xml, '5')

    except Exception as e:
        print(e)

def GenSampleVal(path_image, path_annotate, scale):
    annotation_xml = lxml.etree.fromstring(open(path_annotate).read())
    annotation_xml = parse_xml(annotation_xml)


    width = int(int(annotation_xml['annotation']['size']['width'])*scale)
    height = int(int(annotation_xml['annotation']['size']['height'])*scale)

    scale_bbox_w = height/width
    image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (height, height))

    try:
        objects = annotation_xml['annotation']['object'].copy()
        AddObjectToAnnotete(annotation_xml, 0, 0, 0, 0, True)
        for obj in objects:
            xmin = float(obj['bndbox']['xmin']) * scale_bbox_w
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax']) * scale_bbox_w
            ymax = float(obj['bndbox']['ymax'])
            xmin, ymin, xmax, ymax = RoteteBox(xmin, ymin, xmax, ymax, float(obj['attributes']['attribute']['value']))

            xmin = int(xmin * scale)
            ymin = int(ymin * scale)
            xmax = int(xmax * scale)
            ymax = int(ymax * scale)
            AddObjectToAnnotete(annotation_xml, xmin, ymin, xmax, ymax)

        '''cv2.rectangle(image, (xmin,ymin), (xmax, ymax), 255, thickness=1)
        cv2.imshow('TT', image)
        cv2.waitKey()'''
        SetSize(annotation_xml, height, height)
        SaveSample(save_path_images_val, save_path_annotetions_val, image, annotation_xml, '1')


    except Exception as e:
        print(e)

def GenTrainData(paths_images: list, paths_annotates: list):
    for img, ann in zip(paths_images, paths_annotates):
        print(img)
        GenSample(img, ann, scale)

def GenValData(paths_images: list, paths_annotates: list):
    for img, ann in zip(paths_images, paths_annotates):
        print(img)
        GenSampleVal(img, ann, scale)


def run():
    train = 0
    val = 0
    train_image, train_xml, val_image, val_xml = splitDir(remainder_path)
    train += len(train_image)
    val += len(val_image)
    GenTrainData(train_image, train_xml)
    print('------------------VAL-----------------------------')
    print()
    GenValData(val_image, val_xml)
    print()
    print()

    train_image, train_xml, val_image, val_xml = splitDir(milk365_path)
    train += len(train_image)
    val += len(val_image)
    GenTrainData(train_image, train_xml)
    print('------------------VAL------------------------------')
    print()
    GenValData(val_image, val_xml)
    print()
    print()

    train_image, train_xml, val_image, val_xml = splitDir(jpeg_path, extension='.jpg')
    train += len(train_image)
    val += len(val_image)
    GenTrainData(train_image, train_xml)
    print('-------------------VAL----------------------------')
    print()
    GenValData(val_image, val_xml)
    print()
    print()

    train_image, train_xml, val_image, val_xml = splitDir(milk_200_hard_path)
    train += len(train_image)
    val += len(val_image)
    GenTrainData(train_image, train_xml)
    print('-------------------VAL----------------------------')
    print()
    GenValData(val_image, val_xml)
    print()
    print()

    print('Count train: {}, Count val: {}'.format(train, val))

if __name__ == '__main__':
    run()