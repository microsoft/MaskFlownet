import os
import re
import numpy as np
from timeit import default_timer

def load(prefix, subset, shape = (384, 512), samples = -1, dtype = 4):
	pattern = re.compile(r'{}(\d+)_(\d+).bin'.format(subset) )
	files = [ (int(pattern.match(f).group(1) ), f) for f in os.listdir(prefix) if pattern.match(f) ]
	files = list(sorted(files))
	ret = []
	for _, f in files:
		n = int(pattern.match(f).group(2) )
		load_batch(os.path.join(prefix, f), n, ret, shape, dtype)
		if samples != -1 and len(ret) >= samples:
			ret = ret[: samples]
			break
	return zip(*ret)

def load_batch(fname, n, ret, shape = (384, 512), dtype = 4):
	array_info = [
		(np.uint8, (shape[0], shape[1], 3), shape[0] * shape[1] * 3),
		(np.uint8, (shape[0], shape[1], 3), shape[0] * shape[1] * 3),
		(np.float32 if dtype == 4 else np.float16, (shape[0], shape[1], 2), shape[0] * shape[1] * 2 * dtype)
	]
	with open(fname, 'rb') as f:
		buffer = f.read()
		offset = 0
		for i in range(0, n):
			arr = []
			for dtype, shape, nbytes in array_info:
				result = np.ndarray(shape=shape,
					dtype=dtype,
					buffer=buffer[offset : offset + nbytes],
					order='C')
				offset += nbytes
				arr.append(result)
			if not np.any(np.isnan(arr[-1])):
				ret.append(arr)
