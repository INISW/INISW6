from flask import Blueprint, url_for, render_template, request, redirect, flash
from database.db_handler import DBHandler
import datetime as dt
import json
import os
import cv2
from werkzeug.utils import secure_filename
import os.path as osp
import tempfile
from . import caption

import mmcv

from mmtrack.apis import inference_mot, init_model

from absl import app, flags, logging
from absl.flags import FLAGS
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("len(physical_devices): ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import sys
sys.path.append("C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track_test\\yolov4-deepsort-master")

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

video = Blueprint("video", __name__)

@video.route('/video_tracking', methods=['POST', 'GET'])
def video_tracking(video_id, video_name):

    video_id = int(video_id)
    video_name = video_name['video_name']
    output_video_name = str(video_name[:(len(video_name)-4)])+'_output.mp4'

    print("-" * 30)
    print("video_id: ", video_id)
    print("video_name: ", video_name)
    print("-" * 30)

    # config = 'C:\\Users\\Sihyun\\Desktop\\INISW\\project\\sihyun_track\\mmtracking\\configs\\mot\\deepsort\\deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
    # video_input = os.path.join('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\videos', str(video_name))
    # output=os.path.join('C:\\Users\\Sihyun\\Desktop\\INISW\\project\\INISW_proj\\web\\www\\static\\outputs',output_video_name)
    # checkpoint=None
    # score_thr=0.0
    # device='cuda'
    # show=False
    # backend='cv2'
    # fps=None

    flags.DEFINE_string(f'video', '../static/videos/{video_name}', 'path to input video or set to 0 for webcam')
    flags.DEFINE_string('output', '../static/outputs/{output_video_name}', 'path to output video')

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
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
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

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
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
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
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

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
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (x1, y1, x2, y2)))
            
            print("이것이 좌표")
            print(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            print("Object-ID: ", str(track.track_id))
            print("-" * 30)

            db_handler = DBHandler()

            insert_result = db_handler.insert_video_info(frame_num-1, track.track_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            print("-" * 30)
            print("database result: ", insert_result)
            print("-" * 30)
            # print(frame_num, track.track_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            # insert_into(frame_num-1, track.track_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            
            fps = vid.get(cv2.CAP_PROP_FPS)

            time_in_seconds = frame_num / fps

            minutes = int(time_in_seconds // 60)
            seconds = int(time_in_seconds % 60)
            milliseconds = int((time_in_seconds - int(time_in_seconds)) * 1000)

            print(f"Time for frame {frame_num}: {minutes:02d}:{seconds:02d}.{milliseconds:03d}")


            
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
    # assert output or show
    # # load images
    # if osp.isdir(video_input):
    #    imgs = sorted(
    #        filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
    #               os.listdir(video_input)),
    #        key=lambda x: int(x.split('.')[0]))
    #    IN_VIDEO = False
    # else:
    #    #들어옴
    #    imgs = mmcv.VideoReader(video_input)  #The VideoReader class provides sequence like apis to access video frames. It will internally cache the frames which have been visited
    #    IN_VIDEO = True

    #  # define output
    # if output is not None:
    #    #안들어옴
    #    if output.endswith('.mp4'):
    #        OUT_VIDEO = True
    #        out_dir = tempfile.TemporaryDirectory()
    #        out_path = out_dir.name
    #        _out = output.rsplit(os.sep, 1)
    #        if len(_out) > 1:
    #            os.makedirs(_out[0], exist_ok=True)
    #    else:
    #        OUT_VIDEO = False
    #        out_path = output
    #        os.makedirs(out_path, exist_ok=True)
    # if show or OUT_VIDEO:
    #    if fps is None and IN_VIDEO:
    #        fps = imgs.fps
    #    if not fps:
    #        raise ValueError('Please set the FPS for the output video.')
    #    fps = int(fps)

    # print('\n-------------\n')
    # print('모델빌드\n')

    # # build the model from a config file and a checkpoint file
    # model = init_model(config, checkpoint, device=device)

    # print('\n-------완료------\n')
    # prog_bar = mmcv.ProgressBar(len(imgs))
    # # test and show/save the images
    # for i, img in enumerate(imgs):
    #     frame_id = i
    #     if isinstance(img, str):  # img가 str타입인지-> 아님
    #         img = osp.join(video_input, img)

    #     result = inference_mot(model, img, frame_id, video_id)  #여기만 !

    #     if output is not None:
    #         if IN_VIDEO or OUT_VIDEO:
    #             out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
    #         else:
    #             out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
    #     else:
    #         out_file = None

    #     model.show_result(
    #         frame_id,
    #         video_id,
    #         img,
    #         result,
    #         score_thr=score_thr,
    #         show=show,
    #         wait_time=int(1000. / fps) if fps else 0,
    #         out_file=out_file,
    #         backend=backend)
    #     prog_bar.update()

    # if output and OUT_VIDEO:
    #     print(f'making the output video at {output} with a FPS of {fps}')
    #     mmcv.frames2video(out_path, output, fps=fps, fourcc='avc1')
    #     out_dir.cleanup()

    return json.dumps({'status':'200', 'output_video_name':output_video_name})
    # return caption.
 
@video.route('/test_page/video')
def test_page():
    return redirect('/#keyword')