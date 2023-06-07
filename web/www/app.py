from flask import Flask, Blueprint
from .views import views
import os

def create_app():
    app = Flask(__name__)
    app.register_blueprint(views.views)
    # app.register_blueprint(video.video)
    app.secret_key = 'secret key@#(*@&@(*&#(*@#sfds@'

    # UPLOAD_FOLDER = os.getcwd() + r'\www\static\job_pdf'
    # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    return app