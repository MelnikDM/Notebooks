import os
import argparse
import time
import yaml
import numpy as np

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

    # Определение пути до OUTPUT
    if args['output'] == None:
       DIR_OUTPUT = data_configs['DIR_OUTPUT']
       if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT, exist_ok=True)
    else:
         DIR_OUTPUT = args['output']

     # создаем пустой txt-файл
    start = time.monotonic()
    dir_path = Path(DIR_OUTPUT)
    file_name = 'results.txt'
    file_path = dir_path.joinpath(file_name)

    if dir_path.is_dir():
      if file_path.is_file():
        print('Файл уже существует')
      else:
          with open (dir_path.joinpath(file_name),'w') as f:  
               f.write("Данный файл содержит информацию о возможных дублях.\n\n")
               print('Файл для логгирования создан')
    else:
        print('Директории не существует. Создайте директорию')

    # собираем в список пути до наших изображений и создаем пустой словарь
    print("Вычисляем значения хэшей изображений")
    imagePaths = list(paths.list_images(DIR_INPUT))
    hashes = {}
    # перебираем наши пуи
    for imagePath in imagePaths:
        # загружаем изображение и считаем его хэш
        image = cv2.imread(imagePath)
        h = dhash(image)
        # собираем все файлы с таким же хэшем и добавляем их в словарь (в 'p' у нас, по сути, хранятся дубли)
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p
    # проходимся по всем хэшам, ищем повторяющиеся
    for (h, hashedPaths) in hashes.items():
        if len(hashedPaths) > 1:
           for p in hashedPaths[1:]: # все дубли, что нашли, сохраняем в отдульную папку
              print(f'Похожие изображения {p} перемещены в директорию {DIR_OUTPUT}')
              with open(file_path, 'a', encoding='utf-8') as file:
                   file.write(f'Дубли перемещены в директорию\n{"-"*50}\n   - {p}\n   - {DIR_OUTPUT}\n\n')
              shutil.move(os.path.join(DIR_INPUT, p), DIR_OUTPUT)
    print(f'\nВремя работы скрипта: {time.monotonic() - start}')
              
if __name__ == '__main__':
    args = parse_opt()
    main(args)
