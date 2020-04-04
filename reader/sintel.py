import os
import re
import skimage.io
from functools import lru_cache
import struct
import numpy as np

# ======== PLEASE MODIFY ========
sintel_root = r'path\to\your\Sintel'
split_file = r'path\to\your\Sintel\Sintel_train_val.txt' # r'path\to\your\Sintel\Sintel_train_val_maskflownet.txt'

def list_data(path = None):
	if path is None:
		path = sintel_root
	dataset = dict()
	pattern = re.compile(r'frame_(\d+).png')
	split = np.loadtxt(split_file).astype('i4')
	for part in ('training', 'test'):
		dataset[part] = dict()
		if part == 'training':
			dataset[part + str(1)] = dict()
			dataset[part + str(2)] = dict()
		for subset in ('clean', 'final'):
			dataset[part][subset] = []
			if part == 'training':
				c = 0
				dataset[part + str(1)][subset] = []
				dataset[part + str(2)][subset] = []
			for seq in os.listdir(os.path.join(path, part, subset)):
				frames = os.listdir(os.path.join(path, part, subset, seq))
				frames = list(sorted(map(lambda s: int(pattern.match(s).group(1)),
									 filter(lambda s: pattern.match(s), frames))))
				for i in frames[:-1]:
					entry = [
						os.path.join(path, part, subset, seq, 'frame_{:04d}.png'.format(i)),
						os.path.join(path, part, subset, seq, 'frame_{:04d}.png'.format(i + 1))]
					if part == 'training':
						entry.append(os.path.join(path, part, 'flow', seq, 'frame_{:04d}.flo'.format(i)))
						entry.append(os.path.join(path, part, 'invalid', seq, 'frame_{:04d}.png'.format(i)))
					dataset[part][subset].append(entry)
					if part == 'training':
						dataset[part + str(split[c])][subset].append(entry)
						c = c + 1
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
	flo = Flo(1024, 436)
	if fname.endswith('png'):
		data = skimage.io.imread(fname)
		if data.ndim < 3:
			data = 255 - np.expand_dims(data, -1)
		return data
	elif fname.endswith('flo'):
		return flo.load(fname)

if __name__ == '__main__':
	dataset = list_data()
