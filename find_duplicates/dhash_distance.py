import os
import argparse
import time
import yaml

from PIL import Image, ImageChops
import cv2
from imutils import paths

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


def dhash(path_image, hashSize=8):
    """
    Данная функция попиксельно сравнивает изображения.  

    :param path_img_1: путь до изображения №1.
    :param path_img_2: путь до изображения №2.

    Returns:
        None
    """
    # открываем изображения модулем Image
    gray = cv2.cvtColor(path_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
    diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

    # if similarity <= 10:
    #     print(f'\nВозможно совпадение\n{"-"*50}')
    #     print(f'- {path_img_1}')
    #     print(f'- {path_img_2}')
    #     with open('/content/results.txt', 'a', encoding='utf-8') as file:
    #         file.write(f'Возможно совпадение\n{"-"*50}\n   - {path_img_1}\n   - {path_img_2}\n\n')
    # return


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


    print("[INFO] computing image hashes...")
    imagePaths = list(paths.list_images(DIR_INPUT)
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
	# check to see if there is more than one image with the same hash
	if len(hashedPaths) > 1:
		for p in hashedPaths:
            montage = None
            # load the input image and resize it to a fixed width
            # and heightG
            image = cv2.imread(p)
            image = cv2.resize(image, (150, 150))
            # if our montage is None, initialize it
            if montage is None:
                montage = image
            # otherwise, horizontally stack the images
            else:
                montage = np.hstack([montage, image])
        # show the montage for the hash
        print("[INFO] hash: {}".format(h))
        cv2.imshow("Montage", montage)
        cv2.waitKey(0)


    # start = time.monotonic()
    # if not os.path.exists(DIR_INPUT):
    #     print('[-] Директории не существует')
    #     return
    
    # # создаем пустой txt-файл

    # dir_path = Path('/content')
    # file_name = 'results.txt'
    # file_path = dir_path.joinpath(file_name)

    # if dir_path.is_dir():
    #   if file_path.is_file():
    #     print('Файл уже существует')
    #   else:
    #       with open (dir_path.joinpath(file_name),'w') as f:  
    #            f.write("Данный файл содержит информацию о возможных дублях.\n\n")
    #            print('File was created.')
    # else:
    #     print('Директория не существует. Создайте директорию')

    # # считываем содержимое директории с файлами
    # pictures = os.listdir(DIR_INPUT)

    # check_pic = 0
    # cur_pic = 0

    # # цикл работает, пока индекс проверяемого изображения меньше длины списка всех изображений 
    # while check_pic < len(pictures):
    #     if cur_pic == check_pic:
    #         cur_pic += 1
    #         continue
    #     try:  # сравниваем изображения c текущим. После сравнения увеличиваем число в проверяемом изображении на 1
    #         check_pictures(os.path.join(DIR_INPUT, pictures[cur_pic]), os.path.join(DIR_INPUT, pictures[check_pic]))
    #         check_pic += 1
    #     except IndexError:
    #         break
    # print(f'\nВремя работы скрипта: {time.monotonic() - start}')


if __name__ == '__main__':
    args = parse_opt()
    main(args)
