import argparse
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D

from keras_video_object_detector.library.download_utils import download_file
from keras_video_object_detector.library.video_utils import extract_images
from keras_video_object_detector.library.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, \
    draw_boxes, scale_boxes
from keras_video_object_detector.library.yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, \
    preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = box_class_scores > threshold

    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def yolo_filter_boxes_test():
    with tf.Session() as test_a:
        box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
        box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.shape))
        print("boxes.shape = " + str(boxes.shape))
        print("classes.shape = " + str(classes.shape))


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def iou_test():
    box1 = (2, 1, 4, 3)
    box2 = (1, 2, 3, 4)
    print("iou = " + str(iou(box1, box2)))


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_non_max_suppression_test():
    with tf.Session() as test_b:
        scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
        classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                      iou_threshold=iou_threshold)

    return scores, boxes, classes


def yolo_eval_test():
    with tf.Session() as test_b:
        yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                        tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                        tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
        scores, boxes, classes = yolo_eval(yolo_outputs)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))


class YoloObjectDetector(object):

    def __init__(self):
        self.scores = None
        self.boxes = None
        self.classes = None
        self.yolo_model = None
        self.sess = K.get_session()
        self.class_names = None
        self.anchors = None
        self.image_shape = (720., 1280.)
        self.yolo_outputs = None

    def load_model(self, model_dir_path):
        self.class_names = read_classes(model_dir_path + "/coco_classes.txt")
        self.anchors = read_anchors(model_dir_path + "/yolo_anchors.txt")

        yolo_model_file = model_dir_path + "/yolo.h5"
        yolo_model_file_download_link = 'https://www.dropbox.com/s/krwz5xtpuorah48/yolo.h5?dl=1'
        download_file(yolo_model_file, url_path=yolo_model_file_download_link)
        self.yolo_model = load_model(yolo_model_file)
        print(self.yolo_model.summary())

        # The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial
        # processing and conversion.
        self.yolo_outputs = yolo_head(self.yolo_model.output, self.anchors, len(self.class_names))

        # yolo_outputs gave you all the predicted boxes of yolo_model in the correct format.
        # You're now ready to perform filtering and select only the best boxes.
        self.scores, self.boxes, self.classes = yolo_eval(self.yolo_outputs, self.image_shape)

    def predict_objects_in_image(self, image_file):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

        Arguments:
        sess -- your tensorflow/Keras session containing the YOLO graph
        image_file -- name of an image stored in the "images" folder.

        Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes

        Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
        """

        # Preprocess your image
        image, image_data = preprocess_image(image_file, model_image_size=(608, 608))

        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        out_scores, out_boxes, out_classes = self.sess.run([self.scores, self.boxes, self.classes],
                                                           feed_dict={self.yolo_model.input: image_data,
                                                                      K.learning_phase(): 0
                                                                      })

        return [image, out_scores, out_boxes, out_classes]

    def predict_objects_in_video(self, video_file_path, temp_image_folder=None):
        if temp_image_folder is None:
            temp_image_folder = 'temp_images'

        if not os.path.exists(temp_image_folder):
            os.mkdir(temp_image_folder)

        source_image_folder = temp_image_folder + os.path.sep + 'source'
        target_image_folder = temp_image_folder + os.path.sep + 'output'

        if not os.path.exists(source_image_folder):
            os.mkdir(source_image_folder)

        if not os.path.exists(target_image_folder):
            os.mkdir(target_image_folder)

        extract_images(video_file_path, source_image_folder, image_shape=(1280, 720))

        for f in os.listdir(source_image_folder):
            image_file = source_image_folder + os.path.sep + f

            if os.path.isfile(image_file):
                image, out_scores, out_boxes, out_classes = self.predict_objects_in_image(image_file)
                # Print predictions info
                print('Found {} boxes for {}'.format(len(out_boxes), image_file))
                # Generate colors for drawing bounding boxes.
                colors = generate_colors(self.class_names)
                # Draw bounding boxes on the image file
                draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, colors)
                # Save the predicted bounding box on the image
                output_image_file = target_image_folder + os.path.sep + f
                image.save(output_image_file, quality=90)


def main():
    yolo_filter_boxes_test()
    iou_test()
    yolo_non_max_suppression_test()
    yolo_eval_test()


if __name__ == '__main__':
    main()
