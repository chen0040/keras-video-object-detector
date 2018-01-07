import os

import scipy
from matplotlib.pyplot import imshow

from keras_video_object_detector.library.download_utils import download_file
from keras_video_object_detector.library.yolo import YoloObjectDetector
from keras_video_object_detector.library.yolo_utils import generate_colors, draw_boxes


def main():
    model_dir_path = '../models'

    video_file_path = 'videos/road_video.mp4'
    temp_image_folder = 'frames'

    # download the video file if not exists
    download_file(video_file_path, url_path='https://www.dropbox.com/s/9nlph8ha6g1kxhw/road_video.mp4?dl=1')

    detector = YoloObjectDetector()
    detector.load_model(model_dir_path)

    detector.predict_objects_in_video(video_file_path=video_file_path, temp_image_folder=temp_image_folder)


if __name__ == '__main__':
    main()
