from flask import Blueprint, render_template, request, redirect, flash, url_for
from database.db_handler import DBHandler
import datetime as dt
import json
import os
from werkzeug.utils import secure_filename
from . import video, caption
import pandas as pd


views = Blueprint("views", __name__)

@views.route('/')
def index():
     db_handler = DBHandler()
     truncate_rs = db_handler.truncate()

     if truncate_rs == True:
          print("truncate 성공!")
          return render_template('index.html')
#---------------upload--------------------

@views.route('/drag_upload', methods=['POST', 'GET'])
def drag_upload():
     files = request.files.getlist('files')
     for file in files:
          file_type = file.mimetype
          if file_type != "video/mp4":
               return json.dumps({'status':'error_video', 'message': 'Mp4 파일만 업로드 가능합니다.'})
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
         file_type = files.mimetype

         if file_type != "video/mp4":
              return json.dumps({'status':'error_video', 'message': 'Mp4 파일만 업로드 가능합니다.'})
         files.save(os.path.join('www', 'static', 'videos', secure_filename(files.filename)))
         
         db_handler = DBHandler()
         result = db_handler.insert_video_name(files.filename)

         # 추가된 video_id 불러와서 다시 return 해주기
         video_id = db_handler.get_video_id(files.filename)

         db_handler.close()
         print("video_id: ", video_id[0]['video_id'])
     return json.dumps(video_id[0]['video_id'])

#-----------------------------------------
     
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
          print(type(video_id))
          print(len(video_id))
          print("what is it: ", video_id)
          if video_id == '""':
               return json.dumps({'status':'error_video', 'message': '비디오가 입력되지 않았습니다.'})
          
          input_keyword = request.form['keyword']
          input_keyword = json.loads(input_keyword)
          print("type(video_id): ", type(video_id))
          print("type(input_keyword): ", type(input_keyword))
          print("video_id: ", video_id)
          print("input_keyword: ", input_keyword)
          video_id = int(video_id[1])

          for i in range(len(input_keyword)):
               print(input_keyword[i])

          db = DBHandler()
          
          if len(input_keyword) == 1:
               print("insert_keyword 실행됌!(input_keyword==1)")
               results = db.insert_keyword(input_keyword, video_id)
          else:
               for i in range(len(input_keyword)):
                    print("insert_keyword 실행됌!(input_keyword!=1)")
                    results = db.insert_keyword(input_keyword[i], video_id)
                    print("db_results: ", results)

          video_name = db.get_video_name(video_id)
          # print(results)

          # video_tracking 진행 및 필요한 값들 db에 넣는 작업
          output_video_ = json.loads(video.video_tracking(video_id, video_name, input_keyword, input_type='A'))
          print("output video 나왔고")
          print("video_name: ", output_video_["output_video_name"])

          # 필요한 값들이 db에 들어온 후 captioning 진행
          caption_video_ = json.loads(caption.video_captioning(video_name['video_name'], video_id, input_keyword))

          # keyword 기반 언급된 객체만 반환하는 영상
          final_output_video_ = json.loads(video.video_tracking(video_id, video_name, input_keyword, input_type='B'))

          # video_info 에서 object_id값 기준으로 최초등장 데이터들 가져오기
          db.close()

          db = DBHandler()
          sql_rs = db.get_caption(video_id, input_keyword)
          print("--------------------------------------------------------------")
          print(video_id)
          print(input_keyword)
          print(sql_rs)
          print("--------------------------------------------------------------")
          result = pd.DataFrame(sql_rs)
          db_rs = result.drop_duplicates(subset='object_id',keep='first') 
          db_rs = db_rs.reset_index(drop=True)
          final_data = db_rs.to_dict('records')

          print("final_data: ", final_data)
          db.close()

          # caption file로 저장
          with open('www\\static\\outputs_caption\\'+output_video_['output_video_name'][:-4]+'.txt', 'a', encoding='utf-8') as f:
               f.write("time\t\tObject_id\t\tObject_cap\n\n")
               for i in range(len(final_data)):
                    f.write("["+str(final_data[i]['min'])+':'+str(final_data[i]['sec'])+']')
                    f.write("\t\t")
                    f.write(str(final_data[i]['object_id']))
                    f.write("\t\t")
                    f.write(str(final_data[i]['object_cap']))
                    f.write("\n\n")

          return json.dumps({'status':'success', 'output_video':output_video_["output_video_name"], 'final_data':final_data})
