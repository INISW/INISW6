from flask import Blueprint, render_template, request, redirect, flash, url_for
from database.db_handler import DBHandler
import datetime as dt
import json
import os
from werkzeug.utils import secure_filename

views = Blueprint("views", __name__)

# def jsonResponseFactory(data):
#     '''Return a callable in top of Response'''
#     def callable(response=None, *args, **kwargs):
#         '''Return a response with JSON data from factory context'''
#         return Response(json.dumps({"data":data}), *args, **kwargs)
#     return callable

@views.route('/')
def index():
     return render_template('index.html')

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
