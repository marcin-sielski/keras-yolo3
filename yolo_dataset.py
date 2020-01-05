import sys
import os
import argparse
from yolo import YOLO
from PIL import Image
from uuid import uuid4

def parse_args():

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' +
        YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + 
        YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + 
        YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + 
        str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--score', type=float,
        help='Score to use, default ' + 
        str(YOLO.get_defaults("score"))
    )

    parser.add_argument(
        '--input_path', nargs='?', type=str, required=True, default='',
        help = 'Input dataset path'
    )

    parser.add_argument(
        '--output_path', nargs='?', type=str, required=True, default='',
        help = 'Output dataset path'
    )

    parser.add_argument(
        '--class_name', nargs='?', type=str, required=True, default='',
        help = 'Name of the class used to create dataset'
    )

    parser.add_argument(
        '--width', nargs='?', type=int, required=True,
        help = 'With of the output images'
    )

    parser.add_argument(
        '--height', nargs='?', type=int, required=True,
        help = 'Height of the output images'
    )

    return parser.parse_args()

def create_dataset(**kwargs):

    yolo = YOLO(**dict(kwargs))

    input_path = kwargs.get('input_path', '')
    output_path = kwargs.get('output_path', '')
    class_name = kwargs.get('class_name', '')
    class_names = yolo._get_class()

    if class_name not in class_names:
        yolo.close_session()
        return

    class_name = class_name.replace(' ', '_')
    output_path = output_path + '/' + class_name + '_dataset/'


    try:
        os.makedirs(output_path)
    except OSError:
        pass

    for root, _, files in os.walk(input_path):
        label = os.path.basename(root).lower()
        if len(files) > 0:
            try:
                os.makedirs(output_path + label)
            except:
                pass
        for file in files:
            input_file = root + '/' + file
            
            try:
                image = Image.open(input_file)
            except:
                continue
            else:
                _, images = yolo.detect_image(image)
                for image in images:
                    output_file = output_path + label + '/' + str(uuid4()) + \
                        '.png'
                    image.save(output_file)

    yolo.close_session()

if __name__ == '__main__':
    
    create_dataset(**vars(parse_args()))


