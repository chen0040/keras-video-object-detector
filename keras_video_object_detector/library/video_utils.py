import cv2
import os


def extract_images(video_input_file_path, image_output_dir_path):
    if not os.path.exists(image_output_dir_path):
        os.mkdir(image_output_dir_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            cv2.imwrite(image_output_dir_path + os.path.sep + "frame%d.jpg" % count, image)  # save frame as JPEG file
            count = count + 1