# keras-video-object-detector

Object detector in videos using keras and YOLO

# Usage

### Detect objects in an image using YOLO algorithm

The demo code below can be found in [keras_video_object_detector/demo/detect_objects_in_video.py](keras_video_object_detector/demo/detect_objects_in_video.py)

The demo codes takes in a sample video and output another video that has the detected boxes with class labels 

```python
from keras_video_object_detector.library.download_utils import download_file
from keras_video_object_detector.library.yolo import YoloObjectDetector

model_dir_path = 'keras_video_object_detector/models'

video_file_path = 'keras_video_object_detector/demo/videos/road_video.mp4'
output_video_file_path = 'keras_video_object_detector/demo/videos/predicted_video.mp4'
temp_image_folder = 'frames'

# download the test video file if not exists
download_file(video_file_path, url_path='https://www.dropbox.com/s/9nlph8ha6g1kxhw/road_video.mp4?dl=1')

detector = YoloObjectDetector()
detector.load_model(model_dir_path)

result = detector.detect_objects_in_video(video_file_path=video_file_path,
                                 output_video_path=output_video_file_path,
                                 temp_image_folder=temp_image_folder)
```