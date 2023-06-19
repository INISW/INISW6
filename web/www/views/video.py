from flask import Blueprint
from database.db_handler import DBHandler
import json, os, cv2, time, sys
import pandas as pd
from absl import flags
from absl.flags import FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("len(physical_devices): ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

sys.path.append("C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track_test\\yolov4-deepsort-master")

import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

video = Blueprint("video", __name__)

@video.route('/video_tracking', methods=['POST', 'GET'])
def video_tracking(video_id, video_name, input_keyword, input_type):
    for name in list(flags.FLAGS):
        delattr(flags.FLAGS, name)
    db_handler = DBHandler()
    video_id = int(video_id)
    video_name = video_name['video_name']
    output_video_name = str(video_name[:(len(video_name)-4)])+'_output.mp4'

    flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
    flags.DEFINE_string('weights', 'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track_test\\yolov4-deepsort-master\\checkpoints\\yolov4-tiny-416',
                        'path to weights file')
    flags.DEFINE_integer('size', 416, 'resize images to')
    flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
    flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

    flags.DEFINE_string('output_format', 'H264', 'codec used in VideoWriter when saving video to file')
    flags.DEFINE_float('iou', 0.45, 'iou threshold')
    flags.DEFINE_float('score', 0.50, 'score threshold')
    flags.DEFINE_boolean('dont_show', False, 'dont show video output')
    flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
    flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
    flags.DEFINE_string('video', f'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\videos\\{video_name}', 'path to input video or set to 0 for webcam')
    flags.DEFINE_string('output', f'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\outputs\\{output_video_name}', 'path to output video')

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    print("FLAGS.video: ", FLAGS.video)
    print("FLAGS.output: ", FLAGS.output)

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track_test\\yolov4-deepsort-master\\model_data\\mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        infer = keras.models.load_model(FLAGS.weights)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    total_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    if input_type == 'B':
        db_draw_box = db_handler.draw_box(video_id, input_keyword)
        if len(db_draw_box) == 0:
            return json.dumps({'status': 'fail', 'message': 'No object to track'})
        else:
            print("## 실제로 Tracking이 되야 하는 객체 ID 리스트 ##")
            
            result = pd.DataFrame(db_draw_box)
            db_rs = result.drop_duplicates(subset='object_id',keep='first') 
            result_obj_id = db_rs['object_id'].tolist()
            print("result_obj_id: ", result_obj_id)
            print("-" * 70)
    
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            session.close()
            break
        frame_num +=1
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer.predict(batch_data)

            for value in pred_bbox:
                temp_value = np.expand_dims(value, axis=0)
                boxes = temp_value[:, :, 0:4]
                pred_conf = temp_value[:, :, 4:]

                
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            if input_type == 'A':
                fps = vid.get(cv2.CAP_PROP_FPS)

                time_in_seconds = frame_num / fps

                minutes = int(time_in_seconds // 60)
                seconds = int(time_in_seconds % 60)

                bbox_0 = int(bbox[0]) if int(bbox[0]) > 0 else 0
                bbox_1 = int(bbox[1]) if int(bbox[1]) > 0 else 0
                bbox_2 = int(bbox[2]) if int(bbox[2]) > 0 else 0
                bbox_3 = int(bbox[3]) if int(bbox[3]) > 0 else 0
                
                print("*"*70)
                print("객체 좌표값")
                print(bbox_0, bbox_1, bbox_2, bbox_3)
                insert_result = db_handler.insert_video_info(frame_num-1, track.track_id, bbox_0, bbox_1, bbox_2, bbox_3, video_id, minutes, seconds)
                print("DB Input Result: ", insert_result)
                print("*"*70)
            elif input_type =='B':
                if (int(track.track_id) in result_obj_id):
                    print(int(track.track_id) in result_obj_id)

                    color = colors[int(track.track_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                else:
                    continue

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if FLAGS.output and input_type == 'B':
            print("*"*70)
            print("Tracking이 되어야 하는 Object만 영상에 표시")
            print("*"*70)
            out.write(result)

    return json.dumps({'status':'200', 'output_video_name':output_video_name})