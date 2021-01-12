# tensorflow_lite_video_detection.py

import utils
import time
import tensorflow as tf
import cv2
import numpy as np


class Yolov4TensorFlowLiteModel(object):

    STRIDES = [8, 16, 32]
    ANCHORS = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
    XYSCALE = [1.2, 1.1, 1.05]

    def __init__(self, model_path, names_path, input_size=416, treshold=0.2, iou=0.4):
        self.model_path = model_path
        self.names_path = names_path
        self.input_size = input_size
        self.treshold = treshold
        self.iou = iou
        self.classes = utils.read_class_names(self.names_path)
        self.num_class = len(self.classes)

        self.startup_session()
        print('TensorFlowLiteModel initialization success!')

    def startup_session(self,):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = frame / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32) # Creates a single batch shape: (1, input_size, input_size, 3)

        start_time = time.time()

        self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
        self.interpreter.invoke()
        pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        boxes, pred_conf = utils.filter_boxes(pred[0], pred[1], score_threshold = 0.25,
                                             input_shape = tf.constant([self.input_size, self.input_size]))
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class = 50,
            max_total_size = 50,
            iou_threshold = self.iou,
            score_threshold = self.treshold,
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox, self.classes)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return result


def create_prediction_stream(model_path, names_path, video_source, cycle=0, treshold=0.7, input_size=416):
    '''
    Every cycle frames prediction will be performed. By default every single frame will be transformed.
    '''
    model = Yolov4TensorFlowLiteModel(model_path, names_path, treshold=treshold, input_size=input_size)
    video = cv2.VideoCapture(video_source)
    time.sleep(1)
    
    counter = cycle-1
    while True:
        success, frame = video.read()
        counter += 1
        if counter == cycle:
            if success:
                frame = model.predict(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            cv2.imshow(f'frame_{id(model)}', frame)
            counter = cycle-1
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    video.release()


# def main():
#     model_path = './checkpoints/yolov4-coco-416.tflite'
#     names_path = './data/classes/coco.names'
#     video_source = 'http://208.139.200.133/mjpg/video.mjpg' # 0
#     create_prediction_stream(model_path, names_path, video_source)
#
# if __name__ == '__main__':
#     main()
