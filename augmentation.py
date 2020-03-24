import numpy as np
import math
# from mxnet import nd
from mxnet.gluon import nn

class Augmentation(nn.HybridBlock):
	def __init__(self, angle_range, zoom_range, translation_range, target_shape, orig_shape, batch_size,
		aspect_range = None, relative_angle = 0, relative_scale = (1, 1), relative_translation = 0):
		super().__init__()
		self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range) )
		self._scale_range = zoom_range
		try:
			translation_range = tuple(translation_range)
			if len(translation_range) != 2:
				raise ValueError('expect translation range to have shape [2,], but got {}'.format(translation_range))
		except TypeError:
			translation_range = (-translation_range, translation_range)
		self._translation_range = tuple(map(lambda x : x * 2, translation_range))
		self._target_shape = np.array(target_shape)
		self._orig_shape = np.array(orig_shape)
		self._batch_size = batch_size
		self._unit = np.flip(self._target_shape - 1, axis=0).reshape([2,1]) / np.flip(self._orig_shape - 1, axis=0).reshape([1,2])
		self._relative_scale = relative_scale
		self._relative_angle = tuple(map(lambda x : x / 180 * math.pi * relative_angle, angle_range) )
		self._relative_translation = (-relative_translation * 2, relative_translation * 2)
		self._aspect_range = aspect_range

	def _get_relative_transform(self, F):
		aspect_ratio = (self._target_shape[0] - 1) / (self._target_shape[1] - 1)
		rotation = F.random.uniform(*self._relative_angle, shape=(self._batch_size))
		scale = F.random.uniform(*self._relative_scale, shape=(self._batch_size))
		affine_params = [scale * rotation.cos(), scale * -rotation.sin() * aspect_ratio, F.zeros_like(scale),
						 scale * rotation.sin() / aspect_ratio, scale * rotation.cos(),  F.zeros_like(scale),
						 F.zeros_like(scale), F.zeros_like(scale), F.ones_like(scale)]
		affine = F.reshape(F.stack(*affine_params, axis=1), [0, 3, 3])
		return affine
		
	def hybrid_forward(self, F, img1, img2):
		rotation = F.random.uniform(*self._angle_range, shape=(self._batch_size))
		scale = F.random.uniform(*self._scale_range, shape=(self._batch_size))
		if self._aspect_range is not None:
			aspect_ratio = F.random.uniform(*self._aspect_range, shape=(self._batch_size))
		else:
			aspect_ratio = 1
		pad_x, pad_y = 1 - scale * self._unit[0, 0], 1 - scale * self._unit[1, 1]
		translation_x = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_x + F.random.uniform(*self._translation_range, shape=(self._batch_size))
		translation_y = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_y + F.random.uniform(*self._translation_range, shape=(self._batch_size))
		affine_params = [scale * aspect_ratio * rotation.cos() * self._unit[0, 0], scale * aspect_ratio * -rotation.sin() * self._unit[1, 0], translation_x,
						 scale * rotation.sin() * self._unit[0, 1], scale * rotation.cos() * self._unit[1, 1],  translation_y] 
		affine_params = F.stack(*affine_params, axis=1)

		rel_affine = self._get_relative_transform(F)
		affine_2 = F.reshape(F.batch_dot(F.reshape(affine_params, [0, 2, 3]), rel_affine), [0, 6])
		
		rel_translation = [F.zeros((self._batch_size,)), F.zeros((self._batch_size,)), F.random.uniform(*self._relative_translation, shape=(self._batch_size,)),
			F.zeros((self._batch_size,)), F.zeros((self._batch_size,)), F.random.uniform(*self._relative_translation, shape=(self._batch_size))]
		rel_translation = F.stack(*rel_translation, axis = 1)
		affine_2 = affine_2 + rel_translation

		grid = F.GridGenerator(data=affine_params, transform_type='affine', target_shape=list(self._target_shape))
		img1 = F.BilinearSampler(data=img1, grid=grid)

		grid_2 = F.GridGenerator(data=affine_2, transform_type='affine', target_shape=list(self._target_shape))
		img2 = F.BilinearSampler(data=img2, grid=grid_2)

		return img1, img2
'''
class ChromaticBrightnessAugmentation(nn.HybridBlock):
	def __init__(self, brightness = 0.5, batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.brightness = brightness
		self.batch_size = batch_size
	def hybrid_forward(self, F, img):
		aug = img
		alpha = 1.0 + F.random.uniform(-self.brightness, self.brightness, shape = (self.batch_size, 1, 1, 1))
		aug = F.broadcast_mul(aug, alpha)
		return aug

class ChromaticContrastAugmentation(nn.HybridBlock):
	def __init__(self, contrast = 0.5, batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.contrast = contrast
		self.coefficient = [0.299, 0.587, 0.114]
		self.batch_size = batch_size
	def hybrid_forward(self, F, img):
		aug = img
		alpha = 1.0 + F.random.uniform(-self.contrast, self.contrast, shape = (self.batch_size, 1, 1, 1))
		gray = F.concat(*[img.slice_axis(axis = 1, begin = k, end = k + 1) * self.coefficient[k] for k in range(3)], dim = 1)
		mean = F.mean(gray, keepdims = True, axis = (1, 2, 3))
		gray = 3.0 * (1.0 - alpha) * mean
		aug = F.broadcast_mul(aug, alpha)
		aug = F.broadcast_add(aug, gray)
		return aug
'''
class ChromaticSHAugmentation(nn.HybridBlock):
	def __init__(self, saturation = 0.5, hue = 0.5, batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.saturation = saturation
		self.hue = hue
		self.matrix_yiq = [ [ 0.299,  0.587,  0.114],
							[ 0.596, -0.274, -0.321],
							[ 0.211, -0.523, -0.311]]
		self.matrix_rgb = [ [ 1.   ,  0.956,  0.621],
							[ 1.   , -0.272, -0.647],
							[ 1.   , -1.107,  1.705]]
		self.batch_size = batch_size
	def hybrid_forward(self, F, img):
		aug = img
		alpha = 1.0 + F.random.uniform(-self.saturation, self.saturation, shape = (self.batch_size, 1, 1, 1))
		theta = F.random.uniform(-self.hue * np.pi, self.hue * np.pi, shape = (self.batch_size, 1, 1, 1))
		su = alpha * F.cos(theta)
		sw = alpha * F.sin(theta)
		matrix = [  [0.299 + 0.701 * su + 0.168 * sw, 0.587 - 0.587 * su + 0.330 * sw, 0.114 - 0.114 * su - 0.497 * sw],
					[0.299 - 0.299 * su - 0.328 * sw, 0.587 + 0.413 * su + 0.035 * sw, 0.114 - 0.114 * su + 0.292 * sw],
					[0.299 - 0.300 * su + 1.250 * sw, 0.587 - 0.588 * su - 1.050 * sw, 0.114 + 0.886 * su - 0.203 * sw]]
		aug = F.concat(*[sum([F.broadcast_mul(aug.slice_axis(axis = 1, begin = j, end = j + 1), matrix[i][j]) for j in range(3)]) for i in range(3)], dim = 1)
		return aug
'''
class ChromaticGammaAugmentation(nn.HybridBlock):
	def __init__(self, gamma = (0.7, 1.5), batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.gamma_min, self.gamma_max = gamma
		self.batch_size = batch_size
	def hybrid_forward(self, F, img):
		aug = img
		alpha = F.random.uniform(self.gamma_min, self.gamma_max, shape = (self.batch_size, 1, 1, 1))
		aug = F.broadcast_power(aug, alpha)
		return aug

class ChromaticEigenAugmentation(nn.HybridBlock):
	def __init__(self, batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.batch_size = batch_size
	def hybrid_forward(self, F, img):
		spin_angle = F.random.uniform(low = -np.pi, high = np.pi, shape = (self.batch_size, 3, 1, 1))
		cos_ = [F.cos(spin_angle).slice_axis(axis = 1, begin = k, end = k + 1) for k in range(3)]
		sin_ = [F.sin(spin_angle).slice_axis(axis = 1, begin = k, end = k + 1) for k in range(3)]
		spin_matrix = [ [  cos_[0] * cos_[1], sin_[1] * cos_[2] + sin_[0] * cos_[1] * sin_[2], sin_[1] * sin_[2] - sin_[0] * cos_[1] * cos_[2]],
						[- cos_[0] * sin_[1], cos_[1] * cos_[2] - sin_[0] * sin_[1] * sin_[2], cos_[1] * sin_[2] + sin_[0] * sin_[1] * cos_[2]],
						[  sin_[0]		  ,				   - cos_[0] * sin_[2]		  ,					 cos_[0] * cos_[2]		  ]]
		aug = F.concat(*[sum([F.broadcast_mul(img.slice_axis(axis = 1, begin = j, end = j + 1), spin_matrix[i][j]) for j in range(3)]) for i in range(3)], dim = 1)
		return aug

class ChromaticComposeAugmentation(nn.Block):
	def __init__(self, brightness = 0.2, contrast = 0.5, saturation = 0.5, hue = 0.5, gamma = (0.7, 1.5), batch_size = 1, **kwargs):
		super().__init__(**kwargs)
		self.brightness = brightness
		self.contrast = contrast
		self.saturation = saturation
		self.hue = hue
		self.gamma = gamma
		self.batch_size = batch_size
		self.aug_brightness = ChromaticBrightnessAugmentation(self.brightness, self.batch_size)
		self.aug_contrast = ChromaticContrastAugmentation(self.contrast, self.batch_size)
		self.aug_sh = ChromaticSHAugmentation(self.saturation, self.hue, self.batch_size)
		self.augs = [self.aug_brightness, self.aug_contrast, self.aug_sh]
		self.Gamma = ChromaticGammaAugmentation(self.gamma, self.batch_size)
	
	def forward(self, img1, img2):
		aug = nd.concat(img1, img2, dim = 2)
		augs = random.sample(self.augs, 3)
		for aug_op in augs:
			aug = aug_op(aug)
		aug = aug.clip(0, 1)
		aug = self.Gamma(aug)
		return nd.split(aug, axis = 2, num_outputs = 2)
'''
class ColorAugmentation(nn.HybridBlock):
	def __init__(self, contrast_range, brightness_sigma, channel_range, batch_size, shape, noise_range,
		saturation, hue, gamma_range = None, eigen_aug = False, **kwargs):
		super().__init__(**kwargs)
		self._contrast_range = contrast_range
		self._brightness_sigma = brightness_sigma
		self._channel_range = channel_range
		self._batch_size = batch_size
		self._shape = shape
		self._noise_range = noise_range
		self._gamma_range = gamma_range
		self._eigen_aug = eigen_aug
		self._saturation = saturation
		self._hue = hue

	def hybrid_forward(self, F, img1, img2):
		contrast = F.random.uniform(*self._contrast_range, shape=(self._batch_size, 1, 1, 1)) + 1
		brightness = F.random.normal(scale=self._brightness_sigma, shape=(self._batch_size, 1, 1, 1))
		channel = F.random.uniform(*self._channel_range, shape=(self._batch_size, 3, 1, 1))
		noise_sigma = F.random.uniform(*self._noise_range)
		if self._gamma_range is not None:
			gamma = F.random.uniform(*self._gamma_range, shape = (self._batch_size, 1, 1, 1))

		contrast = contrast.repeat(repeats=3, axis=1)
		brightness = brightness.repeat(repeats=3, axis=1)
		
		alpha = 1.0 + F.random.uniform(-self._saturation, self._saturation, shape = (self._batch_size, 1, 1, 1))
		theta = F.random.uniform(-self._hue * np.pi, self._hue * np.pi, shape = (self._batch_size, 1, 1, 1))
		su = alpha * F.cos(theta)
		sw = alpha * F.sin(theta)
		sh_matrix = [	[0.299 + 0.701 * su + 0.168 * sw, 0.587 - 0.587 * su + 0.330 * sw, 0.114 - 0.114 * su - 0.497 * sw],
									[0.299 - 0.299 * su - 0.328 * sw, 0.587 + 0.413 * su + 0.035 * sw, 0.114 - 0.114 * su + 0.292 * sw],
									[0.299 - 0.300 * su + 1.250 * sw, 0.587 - 0.588 * su - 1.050 * sw, 0.114 + 0.886 * su - 0.203 * sw]]
		
		if self._eigen_aug:
			spin_angle = F.random.uniform(low = -np.pi, high = np.pi, shape = (self._batch_size, 3, 1, 1))
			cos_ = [F.cos(spin_angle).slice_axis(axis = 1, begin = k, end = k + 1) for k in range(3)]
			sin_ = [F.sin(spin_angle).slice_axis(axis = 1, begin = k, end = k + 1) for k in range(3)]
			spin_matrix = [	[	cos_[0] * cos_[1], sin_[1] * cos_[2] + sin_[0] * cos_[1] * sin_[2], sin_[1] * sin_[2] - sin_[0] * cos_[1] * cos_[2]],
											[-cos_[0] * sin_[1], cos_[1] * cos_[2] - sin_[0] * sin_[1] * sin_[2], cos_[1] * sin_[2] + sin_[0] * sin_[1] * cos_[2]],
											[	sin_[0]					 ,-cos_[0] * sin_[2]															,	cos_[0] * cos_[2]															]]

		ret = []
		for img in (img1, img2):
			aug = img
			aug = F.concat(*[sum([F.broadcast_mul(aug.slice_axis(axis = 1, begin = j, end = j + 1), sh_matrix[i][j]) for j in range(3)]) for i in range(3)], dim = 1)
			noise = F.random.normal(scale=1, shape=(self._batch_size, 3) + tuple(self._shape))
			aug = aug + noise * noise_sigma
			mean = F.mean(aug, keepdims=True, axis=(2,3))
			aug = F.broadcast_minus(aug, mean)
			aug = F.broadcast_mul(aug, contrast * channel)
			if self._eigen_aug:
				aug = F.concat(*[sum([F.broadcast_mul(aug.slice_axis(axis = 1, begin = j, end = j + 1), spin_matrix[i][j]) for j in range(3)]) for i in range(3)], dim = 1)
			aug = F.broadcast_add(aug, mean * channel + brightness)
			aug = F.clip(aug, 0, 1)
			if self._gamma_range is not None:
				aug = F.broadcast_power(aug, F.exp(gamma))
			ret.append(aug)

		return ret

class GeometryAugmentation(nn.HybridBlock):
	def __init__(self, angle_range, zoom_range, translation_range, target_shape, orig_shape, batch_size,
		aspect_range = None, relative_angle=None, relative_scale=None, relative_translation=None):
		super().__init__()
		self._angle_range = tuple(map(lambda x : x / 180 * math.pi, angle_range) )
		self._scale_range = zoom_range
		try:
			translation_range = tuple(translation_range)
			if len(translation_range) != 2:
				raise ValueError('expect translation range to have shape [2,], but got {}'.format(translation_range))
		except TypeError:
			translation_range = (-translation_range, translation_range)
		self._translation_range = tuple(map(lambda x : x * 2, translation_range))
		self._target_shape = np.array(target_shape)
		self._orig_shape = np.array(orig_shape)
		self._batch_size = batch_size
		self._unit = np.flip(self._target_shape - 1, axis=0).reshape([2,1]) / np.flip(self._orig_shape - 1, axis=0).reshape([1,2])
		self._relative = relative_angle is not None 
		if self._relative:
			self._relative_scale = relative_scale
			self._relative_angle = tuple(map(lambda x : x / 180 * math.pi * relative_angle, angle_range) )
			self._relative_translation = tuple(map(lambda x: x * relative_translation, self._translation_range)) if relative_translation is not None else None
		self._aspect_range = aspect_range

	def _get_relative_transform(self, F):
		aspect_ratio = (self._target_shape[0] - 1) / (self._target_shape[1] - 1)
		rotation = F.random.uniform(*self._relative_angle, shape=(self._batch_size))
		scale = F.random.uniform(*self._relative_scale, shape=(self._batch_size))
		affine_params = [scale * rotation.cos(), scale * -rotation.sin() * aspect_ratio, F.zeros_like(scale),
						 scale * rotation.sin() / aspect_ratio, scale * rotation.cos(),  F.zeros_like(scale),
						 F.zeros_like(scale), F.zeros_like(scale), F.ones_like(scale)]
		affine = F.reshape(F.stack(*affine_params, axis=1), [0, 3, 3])
		inverse = F.stack(
			rotation.cos() / scale, 
			rotation.sin() / scale,
			-rotation.sin() / scale, 
			rotation.cos() / scale,
			axis=1
		)
		inverse = F.reshape(inverse, [0, 2, 2])
		return affine, inverse
		
	def hybrid_forward(self, F, img1, img2, flow, mask):
		rotation = F.random.uniform(*self._angle_range, shape=(self._batch_size))
		aspect_ratio = F.random.uniform(*self._aspect_range, shape=(self._batch_size)) if self._aspect_range is not None else 1
		scale = F.random.uniform(*self._scale_range, shape=(self._batch_size))
		os = (self._orig_shape[0] - 1, self._orig_shape[1] - 1)
		ts = (self._target_shape[0] - 1, self._target_shape[1] - 1)
		abs_rotation = F.abs(rotation)
		scale = F.minimum(scale, os[1] / (aspect_ratio * (ts[0] * F.sin(abs_rotation) + ts[1] * F.cos(abs_rotation))))
		scale = F.minimum(scale, os[0] / (ts[0] * F.cos(abs_rotation) + ts[1] * F.sin(abs_rotation)))
		pad_x, pad_y = 1 - scale * self._unit[0, 0], 1 - scale * self._unit[1, 1]
		translation_x = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_x + F.random.uniform(*self._translation_range, shape=(self._batch_size))
		translation_y = F.random.uniform(-1, 1, shape=(self._batch_size,)) * pad_y + F.random.uniform(*self._translation_range, shape=(self._batch_size))
		affine_params = [scale * aspect_ratio * rotation.cos() * self._unit[0, 0], scale * aspect_ratio * -rotation.sin() * self._unit[1, 0], translation_x,
						 scale * rotation.sin() * self._unit[0, 1], scale * rotation.cos() * self._unit[1, 1],  translation_y] 
		affine_params = F.stack(*affine_params, axis=1)
		affine_inverse = F.stack(
				rotation.cos() / (scale * aspect_ratio), 
				rotation.sin() / (scale * aspect_ratio),
				-rotation.sin() / scale, 
				rotation.cos() / scale,
				axis=1
			)
		linv = F.reshape(affine_inverse, [0, 2, 2])

		mask = mask.broadcast_like(flow.slice_axis(axis = 1, begin = 0, end = 1))
		rel_affine, rel_inverse = self._get_relative_transform(F)
		affine_2 = F.reshape(F.batch_dot(F.reshape(affine_params, [0, 2, 3]), rel_affine), [0, 6])

		if self._relative_translation is not None:
			rel_translation = F.random.uniform(*self._relative_translation, shape=(self._batch_size, 2, 1, 1))
			rel_scale = F.concat(F.ones([self._batch_size, 1, 1, 1]) * (self._orig_shape[1] - 1) / 2,
						F.ones([self._batch_size, 1, 1, 1]) * (self._orig_shape[0] - 1) / 2, dim=1)
			flow = F.broadcast_minus(flow, rel_translation * rel_scale)

		concat_img = F.concat(img1, mask, F.broadcast_mul(flow, mask), dim=1)
		grid = F.GridGenerator(data=affine_params, transform_type='affine', target_shape=list(self._target_shape))
		force_translation = F.maximum(grid.max(axis=(2, 3), keepdims=True) - 1, 0) + F.minimum(grid.min(axis=(2, 3), keepdims=True) + 1, 0)
		grid = F.broadcast_minus(grid, force_translation)
		grid = grid.clip(-1, 1)
		concat_img = F.BilinearSampler(data=concat_img, grid=grid)
		img1 = F.slice_axis(concat_img, axis=1, begin=0, end=3)
		mask = F.slice_axis(concat_img, axis=1, begin=3, end=4)
		flow = F.slice_axis(concat_img, axis=1, begin=4, end=6)
		flow = F.broadcast_div(flow, F.maximum(mask, 1e-8))

		# relative
		grid_2 = F.GridGenerator(data=affine_2, transform_type='affine', target_shape=list(self._target_shape))
		grid_2 = F.broadcast_minus(grid_2, force_translation)
		if self._relative_translation is not None:
			grid_2 = F.broadcast_add(grid_2, rel_translation)
		img2 = F.BilinearSampler(data=img2, grid=grid_2)
		
		inverse_2 = F.batch_dot(rel_inverse, linv)
		flow = F.reshape_like(F.batch_dot(inverse_2, F.reshape(flow, (0, 0, -3))), flow)

		scale = F.stack(F.ones([self._batch_size]) * (self._target_shape[1] - 1) / 2,
						F.zeros([self._batch_size]),
						F.zeros([self._batch_size]),
						F.ones([self._batch_size]) * (self._target_shape[0] - 1) / 2,
						axis=1) 
		scale = F.reshape(scale, [0, 2, 2])
		I = F.reshape(F.one_hot(F.arange(0, 2), depth=2), [1, 2, 2])
		grid = F.GridGenerator(data=F.reshape(F.one_hot(F.arange(0, 2), depth=3), [1, 6]),
								transform_type='affine',
								target_shape=list(self._target_shape))
		grid = F.reshape(F.repeat(grid, axis=0, repeats=self._batch_size), [0, 0, -3])
		factor = F.batch_dot(F.broadcast_minus(rel_inverse, I), scale)
		flow = flow + F.reshape_like(F.batch_dot(factor, grid), flow)
		return img1, img2, flow, mask
