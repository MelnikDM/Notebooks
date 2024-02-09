import tensorflow as tf
import matplotlib.pyplot as plt
import glob as glob
import numpy as np
import argparse

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
        'o', '--output', 
        default=None,
        help='путь к результатам модели'
    )
    parser.add_argument(
        '-m', '--model', 
        default=mobilenet,
        choices=['mobilenet', 'mobilenetv2'],
        help='Выбор модели'
    )
    args = vars(parser.parse_args())
    return args

# Создаем словарь моделей

models_dict = {
    'mobilenet': tf.keras.applications.mobilenet.MobileNet(weights='imagenet'),
    'mobilenetv2': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
    }


def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images 


image_paths = glob.glob('input/*')
print(f"Found {len(image_paths)} images...")


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
    else:
        DIR_OUTPUT = args['output']
     

    for i, input_image in enumerate(input_images):
        print(f"Processing and classifying on {input_image.split('/')[-1]}")
        # read image using matplotlib to keep an original RGB copy
        orig_image = plt.imread(input_image)
        # read and resize the image
        image = tf.keras.preprocessing.image.load_img(input_image, 
            target_size=(224, 224))
        # add batch dimension
        image = np.expand_dims(image, axis=0)
        # preprocess the image using TensorFlow utils
        image = tf.keras.applications.imagenet_utils.preprocess_input(image)

        # load the model
        model = models_dict[args['model']]
        # forward pass through the model to get the predictions
        predictions = model.predict(image)
        processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(
            preds=predictions
        )
        # print(f"Original predictions: {predictions}")
        print(f"Processed predictions: {processed_preds}")
        print('-'*50)
        
        # create subplot of all images
        plt.subplot(2, 2, i+1)
        plt.imshow(orig_image)
        plt.title(f"{processed_preds[0][0][1]}, {processed_preds[0][0][2] *100:.3f}")
        plt.axis('off')

    plt.savefig(f"{DIR_OUTPUT}/{args['model']}_output.png")
    plt.show()
    plt.close()


if __name__ == '__main__':
    args = parse_opt()
    main(args)