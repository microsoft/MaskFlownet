import datetime
import os
import shutil

class FileLog:
	__timezone__ = 8

	def __init__(self, path, name=None, screen=False):
		if os.path.exists(path):
			shutil.copy(path, path + '.bak')
		self._path = path
		self.f = open(path, 'a')
		self.screen = screen

	@classmethod
	def _localtime(cls):
		return datetime.datetime.utcnow() + datetime.timedelta(hours=cls.__timezone__)

	def _timestamp(self):
		return (self._localtime()).strftime('[%Y/%m/%d %H:%M:%S] ')
	
	def log(self, msg, end='\n'):
		self.f.write(self._timestamp() + msg + end)
		self.f.flush()
		if self.screen:
			print(self._timestamp() + msg + end, end='')

	def close(self):
		self.f.close()
