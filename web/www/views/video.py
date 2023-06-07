from flask import Blueprint, url_for, render_template, request, redirect, flash
from database.db_handler import DBHandler
import datetime as dt
import json
import os
from werkzeug.utils import secure_filename

video = Blueprint("video", __name__)