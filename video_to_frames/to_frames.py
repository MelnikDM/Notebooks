import cv2
import os

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
    

def to_frames(DIR_INPUT, DIR_OUTPUT):
    """
    Данная функция извлекает фреймы из видеофайла.

    :param DIR_INPUT: папка с изображениями.
    :param DIR_OUTPUT: папка, в которую сохраняют изображения.

    Returns:
        None
    """
    cap = cv2.VideoCapture(DIR_INPUT)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Количество фреймов: ", video_length)
    count = 0
    print ("Извлечение фреймов...\n")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(DIR_OUTPUT + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        if (count > (video_length-1)):
            cap.release()
            print ("Извлечение фреймов завершено")
            break


def main(args):

    # Определение пути до INPUT    
    if args['input'] == None:
        DIR_INPUT = data_configs['DIR_INPUT']
    else:
        DIR_INPUT = args['input']

    # Определение пути до OUTPUT
    if args['output'] == None:
        DIR_OUTPUT = data_configs['DIR_OUTPUT']
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)
    else:
        DIR_OUTPUT = args['output']


    to_frames(DIR_INPUT, DIR_OUTPUT)


if __name__ == '__main__':
    args = parse_opt()
    main(args)
