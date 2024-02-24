import os
import argparse
import time
import yaml
import numpy as np

from PIL import Image, ImageChops
import cv2
from imutils import paths
from pathlib import Path
import shutil

def parse_opt():
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', 
        default=None,
        help='Путь до конфигурационного файла'
    )
    parser.add_argument(
        '-i', '--input', 
        help='путь к данным',
    )
    parser.add_argument(
        '-o', '--output', 
        default=None,
        help='путь к результатам модели'
    )
    
    args = vars(parser.parse_args())
    return args


def dhash(path_image, hashSize=8):
    gray = cv2.cvtColor(path_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])

def main(args):

    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)

    # Определение пути до INPUT    
    if args['input'] == None:
        DIR_INPUT = data_configs['DIR_INPUT']
    else:
        DIR_INPUT = args['input']

    if args['output'] == None:
       DIR_OUTPUT = data_configs['DIR_OUTPUT']
       if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT, exist_ok=True)
    else:
         DIR_OUTPUT = args['output']


    dir_path = Path('DIR_OUTPUT')
    file_name = 'results.txt'
    file_path = dir_path.joinpath(file_name)

    if dir_path.is_dir():
      if file_path.is_file():
        print('Файл уже существует')
      else:
          with open (dir_path.joinpath(file_name),'w') as f:  
               f.write("Данный файл содержит информацию о возможных дублях.\n\n")
               print('Файл с log создан')
    else:
        print('Директории не существует. Создайте директорию')


    print("Вычисляем значения хэшей изображений")
    imagePaths = list(paths.list_images(DIR_INPUT))
    hashes = {}
    # loop over our image paths
    for imagePath in imagePaths:
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)
        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

    for (h, hashedPaths) in hashes.items():
        if len(hashedPaths) > 1:
           for p in hashedPaths[1:]:
              print(f'Похожие изображения {p} перемещены в директорию {DIR_OUTPUT}')
              with open('/content/output/results.txt', 'a', encoding='utf-8') as file:
                   file.write(f'Дубли перемещены в директорию\n{"-"*50}\n   - {p}\n   - {DIR_OUTPUT}\n\n')
              shutil.move(os.path.join(DIR_INPUT, p), DIR_OUTPUT)
              
if __name__ == '__main__':
    args = parse_opt()
    main(args)
