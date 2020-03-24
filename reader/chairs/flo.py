import struct
import numpy as np

__floheader__ = b'PIEH\x00\x02\x00\x00\x80\x01\x00\x00'
__floec1__ = float(202021.25)
__floec2__ = int(512)
__floec3__ = int(384)
__floheaderlen__ = len(__floheader__)
__flow__ = 512
__floh__ = 384
__floshape__ = [__floh__, __flow__, 2]

__floec1m__, __floec2m__, __floec3m__ = struct.unpack('fii', __floheader__)
if __floec1m__ != __floec1__ or __floec2m__ != __floec2__ or __floec3m__ != __floec3__:
	raise Exception('Expect machine to be LE.')

def load(file):
	with open(file, 'rb') as fp:
		if fp.read(__floheaderlen__) != __floheader__:
			raise Exception('Bad flow header: ' + file)
		result = np.ndarray(shape=__floshape__,
			dtype=np.float32,
			buffer=fp.read(),
			order='C')
		return result
