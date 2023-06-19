from flask import Flask
from .views import views, video, caption

def create_app():
    app = Flask(__name__)
    app.register_blueprint(views.views)
    app.register_blueprint(video.video)
    app.register_blueprint(caption.caption)
    app.secret_key = 'secret key@#(*@&@(*&#(*@#sfds@'

    return app