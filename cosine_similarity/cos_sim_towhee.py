import pandas as pd
import numpy as np

import argparse
import yaml
import os

from keras.preprocessing import image as kimage
from keras.applications import MobileNetV2
from keras.applications.mobilenet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from towhee import AutoPipes
from PIL import Image
from random import randint

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


# def process_all_image(test_images):
#     """
#     Данная функция отвечает за препроцессинг входных данных. Во время предварительной обработки
#     размер изображения изменяется, а значения пикселей нормализуются до диапазона [-1, 1]
    
#     :param test_images: изображения, которые необходимо обработать.
    
#     Returns:
#         img_dict: Словарь следующей структуры: {№ изображения: тензор}.
    
#     """
#     for mix_image in test_images:
#         image = kimage.load_img(mix_image, target_size = (224,224))
#         image = preprocess_input(np.expand_dims(kimage.img_to_array(image), axis = 0))
#         num = mix_image.split('/')[-1].split('.')[0]
#         img_dict[num] = image
    
#     return img_dict


def flatten_pixels(img_path):
    """
    Данная функция отвечает за препроцессинг входных данных с помощью бибилиотеки towhee.
    Используя towhee мы преобразуем наши изображения в вектора (эмбеддинги)
    
    :param img_path: путь до изображений.
    
    Returns:
        image_df: Датафрейм со столбцами: img_path, embedding_tt.
    
    """
    image_df = pd.DataFrame(img_path, columns=['img_path'])
    print(image_df.shape)
    p = AutoPipes.pipeline('image-embedding')
    image_df['embedding_tt'] = image_df['img_path'].apply(lambda x: np.squeeze(p(x).get()))

    return image_df


def plot_similar(df, embedding_col, dir_out, query_index, k_neighbors=3):
    """
    Данная функция отвечает за:
     - подсчет косинусного сходства;
     - определяет количество отображаемых ближайших соседей;
     - сохраняет результаты в .csv и в .png.
    
    :param df: путь до изображений;
    :param embedding_col: целевой столбец с эмбеддингами;
    :param dir_out: путь до output;
    :param query_index: целевое изображение;
    :param k_neighbors: количество соседей.
            
    """

    # Считаем косинусное сходство и сохраняем все в .csv
    cos_similarity = cosine_similarity([df[embedding_col][query_index]], df[embedding_col].values.tolist())[0]
    cosine_dataframe = pd.DataFrame(cos_similarity)
    cosine_dataframe.to_csv(f'{dir_out}/cos_df.csv', sep=';')
    print()
    print(f"Результаты сохранены в {dir_out}")
    print()

    # Находим ближайших соседей
    k = k_neighbors+1
    img_list = np.argsort(-cos_similarity)[:k]
    img_list = img_list[img_list != query_index]

    # Отображение изображения из запроса
    img = Image.open(df['img_path'][query_index]).convert('RGB')
    plt.imshow(img)
    plt.title('Изображение из запроса')

    # Отображение ближайших соседей
    fig = plt.figure(figsize=(20,4))
    plt.suptitle('Похожие изображения')
    for i, image in enumerate(img_list):
        plt.subplot(1, len(img_list), i+1)
        img = Image.open(df['img_path'][image]).convert('RGB')
        plt.imshow(img)
        plt.title(f'Cosine similiarity: {cos_similarity[image]:.3f}')
    plt.savefig(f"{dir_out}/output.png")
    plt.show()
    plt.close()
    plt.tight_layout()


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
    print()

    # Определение пути до OUTPUT
    if args['output'] == None:
       DIR_OUTPUT = data_configs['DIR_OUTPUT']
       if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT, exist_ok=True)
    else:
         DIR_OUTPUT = args['output']

    # Препроцессинг изображений
    image_df = flatten_pixels(input_images)

    a = int(input("Введите номер изображения: "))
    print()
    b = int(input("Введите количество соседей : "))
    print()

    plot_similar(df=image_df,
                embedding_col='embedding_tt',
                dir_out = DIR_OUTPUT,
                query_index=a,
                k_neighbors=b)


if __name__ == '__main__':
    args = parse_opt()
    main(args)