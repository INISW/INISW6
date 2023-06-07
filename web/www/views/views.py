from flask import Blueprint, render_template, request, redirect, flash, url_for
from database.db_handler import DBHandler
import datetime as dt
import json
import os
from werkzeug.utils import secure_filename

import os.path as osp
import tempfile

import mmcv

from mmtrack.apis import inference_mot, init_model

views = Blueprint("views", __name__)

# def jsonResponseFactory(data):
#     '''Return a callable in top of Response'''
#     def callable(response=None, *args, **kwargs):
#         '''Return a response with JSON data from factory context'''
#         return Response(json.dumps({"data":data}), *args, **kwargs)
#     return callable

@views.route('/')
def index():
     # return render_template('index.html')
     return json.dumps({'status':'200'})

#-------------------------------------------

@views.route('/drag_upload', methods=['POST', 'GET'])
def drag_upload():
     files = request.files.getlist('files')
     for file in files:
          fn = secure_filename(file.filename)
          file.save(os.path.join('www', 'static', 'videos', fn))  # replace FILES_DIR with your own directory
          db_handler = DBHandler()
          result = db_handler.insert_video_name(file.filename)

          # 추가된 video_id 불러와서 다시 return 해주기
          video_id = db_handler.get_video_id(file.filename)
          db_handler.close()

          print("video_id: ", video_id[0]['video_id'])
     return json.dumps(video_id[0]['video_id'])
    
@views.route('/upload', methods=['POST', 'GET'])
def upload():
     if request.method == 'POST':
         files = request.files['video']
         files.save(os.path.join('www', 'static', 'videos', secure_filename(files.filename)))
         
         db_handler = DBHandler()
         result = db_handler.insert_video_name(files.filename)

         db_handler.close()

     flash('업로드가 완료되었습니다.', 'success')
     return redirect('/#fold')
     
@views.route('/get_video_id', methods=['POST', 'GET'])
def get_video_id():
     if request.method == 'POST':
          filename = request.form['filename']

          db = DBHandler()
          results = db.get_video_id(filename)
          print("results: ", results)
          db.close()

     return json.dumps({'status':'200', 'video_id': results[0]['video_id']})

@views.route('/keyword', methods=['POST', 'GET'])
def keyword():
     if request.method == 'POST':
          video_id = request.form['video_id']
          input_keyword = request.form['keyword']
          input_keyword = json.loads(input_keyword)
          print(type(video_id))
          print(type(input_keyword))
          print(video_id)
          print(input_keyword)
          video_id = int(video_id[1])

          for i in range(len(input_keyword)):
               print(input_keyword[i])

          


          db = DBHandler()
          
          for i in range(len(input_keyword)):
               results = db.insert_keyword(input_keyword[i], video_id)
               print("db_results: ", results)

          db.close()

          

          print(results)

          return json.dumps({'status':'200'})


@views.route('/test')
def test():
     config = '/Users/dahyeon/Documents/repos/mmtracking/mmdetection-2.28.2/mmtracking-0.14.0/configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py'
     video_input = '/Users/dahyeon/Documents/repos/mmtracking/mmdetection-2.28.2/mmtracking-0.14.0/demo/demo.mp4'
     output='ouputs/demo_output.mp4'
     checkpoint=None
     score_thr=0.0
     device='cpu'
     show=False
     backend='cv2'
     fps=None

     assert output or show
     # load images
     if osp.isdir(video_input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(video_input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
     else:
        #들어옴
        imgs = mmcv.VideoReader(video_input)  #The VideoReader class provides sequence like apis to access video frames. It will internally cache the frames which have been visited
        IN_VIDEO = True

        # print('\n-----\n')
        # print("obtain basic information")

        # print(len(imgs))
        # print(imgs.width, imgs.height, imgs.resolution, imgs.fps)
        #1920 1080 (1920, 1080) 3.0

     # define output
     if output is not None:
        #안들어옴
        if output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = output
            os.makedirs(out_path, exist_ok=True)

     if show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)
        # print("\n********************\n")
        # print(fps)
        # sys.exit()

     print('\n-------------\n')
     print('모델빌드\n')

     # build the model from a config file and a checkpoint file
     model = init_model(config, checkpoint, device=device)

     print('\n-------완료------\n')
     prog_bar = mmcv.ProgressBar(len(imgs))
     # test and show/save the images
     for i, img in enumerate(imgs):
          frame_id = i
          if isinstance(img, str):  # img가 str타입인지-> 아님
               img = osp.join(video_input, img)

          print('\n-------여기1------\n')
          # print(img.shape) = (1080, 1920, 3)

          result = inference_mot(model, img, frame_id)  #여기만 !
          
          # {'det_bboxes': [array([], shape=(0, 5), dtype=float32)], 'track_bboxes': [array([], shape=(0, 6), dtype=float32)]}
          print('\n-------여기2------\n')

          if output is not None:
               if IN_VIDEO or OUT_VIDEO:
                    out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
               else:
                    out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
          else:
               out_file = None

          print('\n**for문**\n')

          model.show_result(
               frame_id,
               img,
               result,
               score_thr=score_thr,
               show=show,
               wait_time=int(1000. / fps) if fps else 0,
               out_file=out_file,
               backend=backend)
          prog_bar.update()

     if output and OUT_VIDEO:
          print(f'making the output video at {output} with a FPS of {fps}')
          mmcv.frames2video(out_path, output, fps=fps, fourcc='mp4v')
          out_dir.cleanup()



     return json.dumps({'status':'200'})