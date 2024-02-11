import pandas as pd
import numpy as np

import argparse
import yaml
import os

from keras.preprocessing import image as kimage
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from glob import glob

img_dict = dict()

# Создаем парсер

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


# Создаем загрузчик изображений

def collect_all_images(dir_test):
    """
    Данная функция возвращает список путей до изображений.

    :param dir_test: папка с изображениями.

    Returns:
        test_images: Список с путями.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images 



def process_all_image(test_images):
    """
    Данная функция отвечает за препроцессинг входных данных. Во время предварительной обработки
    размер изображения изменяется, а значения пикселей нормализуются до диапазона [-1, 1]/
    
    :param test_images: изображения, которые необходимо обработать.
    
    Returns:
        img_dict: Словарь следующей структуры: {№ изображения: тензор}.
    
    """
    for mix_image in test_images:
        image = kimage.load_img(mix_image, target_size = (224,224))
        image = preprocess_input(np.expand_dims(kimage.img_to_array(image), axis = 0))
        num = mix_image.split('/')[-1].split('.')[0]
        img_dict[num] = image
    
    return img_dict

def main(args):

    np.random.seed(42)

    # Загрузка файла конфигурации
    data_configs = None
    if args['config'] is not None:
        with open(args['config']) as file:
            data_configs = yaml.safe_load(file)

    # Определение пути до INPUT    
    if args['input'] == None:
        DIR_INPUT = data_configs['DIR_INPUT']
        input_images = collect_all_images(DIR_INPUT)
    else:
        DIR_INPUT = args['input']
        input_images = collect_all_images(DIR_INPUT)
    print(f"Количество изображений: {len(input_images)}")

    # Определение пути до OUTPUT
    if args['output'] == None:
        DIR_OUTPUT = data_configs['DIR_OUTPUT']
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)
    else:
        DIR_OUTPUT = args['output']


    img_dict = process_all_image(input_images)

    mobilenetv2_model = MobileNetV2(include_top = False, weights = 'imagenet')

    
   
    #initialize the matrix (62720)
    images_matrix = np.zeros([len(img_dict.values()), 62720])
    for i, (num, image) in enumerate(img_dict.items()):
        images_matrix[i, :] = mobilenetv2_model.predict(image).ravel()

    
    cos_similarity = cosine_similarity(images_matrix)

    cosine_dataframe = pd.DataFrame(cos_similarity)
    print(cosine_dataframe)
    print("Shape of the cosine dataframe is:", cosine_dataframe.shape)
    print()
    print()

    product_info = cosine_dataframe.iloc[6].values
    similar_images_index = np.argsort(-product_info)[:3]
    print("Index of similar images are:", similar_images_index)
    print(sorted(-product_info)[: 3])

    image_path = glob(f"{DIR_INPUT}/*.jpg")

    img_list = [image_path[similar_images_index[0]], image_path[similar_images_index[1]], image_path[similar_images_index[2]]]

    for i, image in enumerate(img_list):
        img = cv2.imread(img_list[i])

        plt.subplot(2, 3, i+1)
        plt.imshow(img)

    plt.savefig(f"{DIR_OUTPUT}/output.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    args = parse_opt()
    main(args)