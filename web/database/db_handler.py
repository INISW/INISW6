from distutils.command.clean import clean
import pymysql
import configparser
import json
import time
from itertools import islice

class DBHandler:
	
	def __init__(self):
		# parse 'config.ini' file automatically
		# please find 'config.ini' file or create it and place it in 'root' folder
		# and enter your mysql or mariadb connection information
		parser = configparser.ConfigParser()
		parser.read('config.ini')

		DB_props = parser["DATABASE"]
		DB_HOST = DB_props['HOST']
		DB_PORT = DB_props.getint('PORT')
		DB_NAME = DB_props['NAME']
		DB_USER = DB_props['USER']
		DB_PW = DB_props['PW']
		self.__conn = pymysql.connect(
						host=DB_HOST,
						port=DB_PORT,
						database=DB_NAME,
						user=DB_USER,
						password=DB_PW)
	

	def get_video_info(self):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			SELECT * FROM video_info
			'''

		cursor.execute(sql)
		sql_rs = cursor.fetchall()

		return sql_rs
	
	def get_video_id(self, video_name):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			SELECT video_id FROM video_keyword
			WHERE video_name = %s
		'''

		cursor.execute(sql, video_name)
		sql_rs = cursor.fetchall()

		return sql_rs

	def insert_video_name(self, video_name):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			INSERT INTO video_keyword VALUES(%s, %s, %s)
			'''
		try:
			cursor.execute(sql, (None, video_name, None))
		except pymysql.err.DataError:
			return False
		self.__conn.commit()
		return True
	
	def insert_keyword(self, keyword, video_id):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			INSERT INTO keyword_list VALUES(%s, %s, %s)
		'''
		try:
			cursor.execute(sql, (None, keyword, video_id))
		except pymysql.err.DataError:
			return False
		self.__conn.commit()
		return True

	def close(self):
		self.__conn.close()
	
db = DBHandler()