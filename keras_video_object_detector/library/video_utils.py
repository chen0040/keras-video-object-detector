import cv2
import os


def extract_images(video_input_file_path, image_output_dir_path, image_shape=None, frames_per_second=None):
    if frames_per_second is None:
        frames_per_second = 20
    if not os.path.exists(image_output_dir_path):
        os.mkdir(image_output_dir_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 // frames_per_second))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            if image_shape is not None:
                image = cv2.resize(image, image_shape, interpolation=cv2.INTER_AREA)
            image_file = image_output_dir_path + os.path.sep + "frame%4d.jpg" % count
            cv2.imwrite(image_file, image)  # save frame as JPEG file
            print('extracting ' + image_file)
            count = count + 1