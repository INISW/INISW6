from distutils.command.clean import clean
import pymysql
import configparser
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
			SELECT * FROM video_info_copy1
			'''

		cursor.execute(sql)
		sql_rs = cursor.fetchall()

		return sql_rs
	
	def get_video_id(self, video_name):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			SELECT video_id FROM video_list
			WHERE video_name = %s
		'''

		cursor.execute(sql, video_name)
		sql_rs = cursor.fetchall()

		return sql_rs
	
	def get_video_name(self, video_id):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			SELECT video_name FROM video_list
			WHERE video_id = %s
		'''

		cursor.execute(sql, video_id)
		sql_rs = cursor.fetchone()

		return sql_rs

	def insert_video_name(self, video_name):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			INSERT INTO video_list VALUES(%s, %s)
			'''
		try:
			cursor.execute(sql, (None, video_name))
		except pymysql.err.DataError:
			return False
		self.__conn.commit()
		return True
	
	def insert_video_info(self, frame_id, object_id, x1, y1, x2, y2, video_id, min, sec):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			INSERT INTO video_info_copy1(frame_id, object_id, x1, y1, x2, y2, video_id, min, sec) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
		'''
		try:
			cursor.execute(sql, (frame_id, object_id, x1, y1, x2, y2, video_id, min, sec))

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
	
	#----------caption-------------
	def db_to_dataframe(self):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			SELECT frame_id, object_id, x1, y1, x2, y2, video_id, min, sec FROM video_info_copy1
		'''
		cursor.execute(sql)
		sql_rs = cursor.fetchall()

		print("-왜 문제가 생긴건데?" * 40)
		print(sql_rs)
		print("-" * 40)
		return sql_rs
	
	def savecaption(self, data, captions):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql = '''
			UPDATE video_info_copy1 SET object_cap = %s WHERE object_id = %s
		'''

		# print("data.iterrows: ", data.iterrows())
		# print("captions: ", captions)

		try:
			for index, row in data.iterrows(): # for문으로 object_id, caption matching 하여 올리기
				cursor.execute(sql, (captions[index], int(row.object_id)))
		except pymysql.err.DataError:
			return False

		self.__conn.commit()

		return True
	
	def get_caption(self, video_id, input_keyword):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		try:
			if (len(input_keyword) == 0):
				sql = '''
						SELECT frame_id, object_id, object_cap, min, sec
						FROM video_info_copy1
						WHERE video_id = %s
						'''
			else:
				sql = '''
						SELECT frame_id, object_id, object_cap, min, sec
						FROM video_info_copy1
						WHERE video_id = %s AND
						object_cap REGEXP "^'''
				
				for keyword in input_keyword:
					regexp_com_start = "(?=.*"
					regexp_com_end = ")"
					keyword_com = regexp_com_start + keyword + regexp_com_end

					sql = sql + keyword_com

				sql = sql + '.*$"'

			print("this is get_caption complete sql command", sql)
			cursor.execute(sql, video_id)

		except pymysql.err.DataError:
			return False

		sql_rs = cursor.fetchall()

		print(sql_rs)
		
		return sql_rs
	
	def draw_box(self, video_id, input_keyword):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		try:
			if (len(input_keyword) == 0):
				sql = '''
						SELECT frame_id, object_id, x1, y1, x2, y2
						FROM video_info_copy1
						WHERE video_id = %s
						'''
			else:
				sql = '''
					SELECT frame_id, object_id, x1, y1, x2, y2
					FROM video_info_copy1
					WHERE video_id = %s AND
					object_cap REGEXP "^'''
			
				for keyword in input_keyword:
					regexp_com_start = "(?=.*"
					regexp_com_end = ")"
					keyword_com = regexp_com_start + keyword + regexp_com_end

					sql = sql + keyword_com

				sql = sql + '.*$"'
			

			print("this is draw_box complete sql command", sql)
			cursor.execute(sql, video_id)

		except pymysql.err.DataError:
			return False

		sql_rs = cursor.fetchall()
		
		return sql_rs
	#-------truncate-------
	def truncate(self):
		cursor = self.__conn.cursor(pymysql.cursors.DictCursor)
		sql1 = '''
			truncate table video_info_copy1
		'''
		sql2 = '''
			truncate table video_list
		'''
		sql3 = '''
			truncate table keyword_list
		'''

		try:
			cursor.execute(sql1)
			cursor.execute(sql2)
			cursor.execute(sql3)
		except pymysql.err.DataError:
			return False
		
		self.__conn.commit()
		return True

	def close(self):
		self.__conn.close()
	
db = DBHandler()