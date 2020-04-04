import os
import numpy as np
from functools import lru_cache
import struct

# ======== PLEASE MODIFY ========
things3d_root = r'path\to\your\FlyingThings3D_subset'

def list_data(path = None, sub_type = 'clean'):
	if path is None:
		path = things3d_root
	parts = ('train', ) # 'val')
	sub_types = (sub_type,)
	if sub_type == 'mixed':
		sub_types = ('clean', 'final')
	orients = ('into_future', 'into_past')
	cameras = ('left', 'right')
	dataset = dict()
	dataset['image_0'] = []
	dataset['image_1'] = []
	dataset['flow'] = []
	for part in parts:
		for sub_type in sub_types:
			for camera in cameras:
				for orient in orients:
					flow_ind = 1 if orient == 'into_future' else - 1
					path_image = os.path.join(path, part, 'image_' + sub_type, camera)
					path_flow = os.path.join(path, part, 'flow', camera, orient)
					dirs_flow = os.listdir(path_flow)
					for dir_flow in dirs_flow:
						dataset['flow'].append(os.path.join(path_flow, dir_flow))
						dataset['image_0'].append(os.path.join(path_image, dir_flow.replace('flo', 'png')))
						ind = int(dir_flow[-11:-4])
						dataset['image_1'].append(os.path.join(path_image, dir_flow.replace('flo', 'png').replace('%07d' % ind, '%07d' % (ind + flow_ind))))

	return dataset

class Flo:
	def __init__(self, w, h):
		self.__floec1__ = float(202021.25)
		self.__floec2__ = int(w)
		self.__floec3__ = int(h)
		self.__floheader__ = struct.pack('fii', self.__floec1__, self.__floec2__, self.__floec3__)
		self.__floheaderlen__ = len(self.__floheader__)
		self.__flow__ = w
		self.__floh__ = h
		self.__floshape__ = [self.__floh__, self.__flow__, 2]

		if self.__floheader__[:4] != b'PIEH':
			raise Exception('Expect machine to be LE.')

	def load(self, file):
		with open(file, 'rb') as fp:
			if fp.read(self.__floheaderlen__) != self.__floheader__:
				raise Exception('Bad flow header: ' + file)
			result = np.ndarray(shape=self.__floshape__,
								dtype=np.float32,
								buffer=fp.read(),
								order='C')
			return result

	def save(self, arr, fname):
		with open(fname, 'wb') as fp:
			fp.write(self.__floheader__)
			fp.write(arr.astype(np.float32).tobytes())

@lru_cache(maxsize=None)
def load(fname):
	flo = Flo(960, 540)
	if fname.endswith('flo'):
		return flo.load(fname)

if __name__ == '__main__':
	dataset = list_data()
	print(len(dataset['flow']))
	print(dataset['flow'][-1])