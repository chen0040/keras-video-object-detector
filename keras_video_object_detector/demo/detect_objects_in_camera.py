import cv2
from keras_video_object_detector.library.yolo import YoloObjectDetector


def main():
    model_dir_path = '../models'

    detector = YoloObjectDetector()
    detector.load_model(model_dir_path)

    camera = cv2.VideoCapture(0)

    detector.detect_objects_in_camera(camera=camera)

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
