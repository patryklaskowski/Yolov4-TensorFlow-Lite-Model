from TensorFlowLiteModel import create_prediction_stream

def main():
    model_path = './checkpoints/yolov4-coco-416.tflite'
    names_path = './data/classes/coco.names'
    video_source = 0 #'http://208.139.200.133/mjpg/video.mjpg' # 0
    create_prediction_stream(model_path, names_path, video_source)

if __name__ == '__main__':
    main()
