import numpy as np

__ppmheader__ = b'P6 512 384 255\n'
__ppmheaderlen__ = len(__ppmheader__)
__ppmw__ = 512
__ppmh__ = 384
__ppmshape__ = [__ppmh__, __ppmw__, 3]

def load(file):
	with open(file, 'rb') as fp:
		if fp.read(__ppmheaderlen__) != __ppmheader__:
			raise Exception('Bad ppm header: ' + file)
		result = np.ndarray(shape=__ppmshape__,
			dtype=np.uint8,
			buffer=fp.read(),
			order='C')
		return result
