import os
import argparse
import time
import yaml

from PIL import Image, ImageChops
from pathlib import Path
import imagehash

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
    
    args = vars(parser.parse_args())
    return args


def check_pictures(path_img_1, path_img_2):
    """
    Данная функция попиксельно сравнивает изображения.  

    :param path_img_1: путь до изображения №1.
    :param path_img_2: путь до изображения №2.

    Returns:
        None
    """
    # открываем изображения модулем Image
    img_1 = Image.open(path_img_1)
    img_2 = Image.open(path_img_2)
    
    # проводим изображения к одному размеру
    img_1.thumbnail((640, 640))
    img_2.thumbnail((640, 640))

    image_one_hash = imagehash.whash(img_1)
    image_two_hash = imagehash.whash(img_2)
    similarity = image_one_hash - image_two_hash

    if similarity <= 10:
        print(f'\nВозможно совпадение\n{"-"*50}')
        print(f'- {path_img_1}')
        print(f'- {path_img_2}')
        with open('/content/results.txt', 'a', encoding='utf-8') as file:
            file.write(f'Возможно совпадение\n{"-"*50}\n   - {path_img_1}\n   - {path_img_2}\n\n')
    return


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


    start = time.monotonic()
    if not os.path.exists(DIR_INPUT):
        print('[-] Директории не существует')
        return
    
    # создаем пустой txt-файл

    dir_path = Path('/content')
    file_name = 'results.txt'
    file_path = dir_path.joinpath(file_name)

    if dir_path.is_dir():
      if file_path.is_file():
        print('Файл уже существует')
      else:
          with open (dir_path.joinpath(file_name),'w') as f:  
               f.write("Данный файл содержит информацию о возможных дублях.\n\n")
               print('File was created.')
    else:
        print('Директория не существует. Создайте директорию')

    # считываем содержимое директории с файлами
    pictures = os.listdir(DIR_INPUT)

    check_pic = 0
    cur_pic = 0

    # цикл работает, пока индекс проверяемого изображения меньше длины списка всех изображений 
    while check_pic < len(pictures):
        if cur_pic == check_pic:
            cur_pic += 1
            continue
        try:  # сравниваем изображения c текущим. После сравнения увеличиваем число в проверяемом изображении на 1
            check_pictures(os.path.join(DIR_INPUT, pictures[cur_pic]), os.path.join(DIR_INPUT, pictures[check_pic]))
            check_pic += 1
        except IndexError:
            break
    print(f'\nВремя работы скрипта: {time.monotonic() - start}')


if __name__ == '__main__':
    args = parse_opt()
    main(args)
