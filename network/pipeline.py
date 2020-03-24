import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd

from .MaskFlownet import *
from .config import Reader
from .layer import Reconstruction2D, Reconstruction2DSmooth

def build_network(name):
	return eval(name)

def get_coords(img):
	shape = img.shape
	range_x = nd.arange(shape[2], ctx = img.context).reshape(shape = (1, 1, -1, 1)).tile(reps = (shape[0], 1, 1, shape[3]))
	range_y = nd.arange(shape[3], ctx = img.context).reshape(shape = (1, 1, 1, -1)).tile(reps = (shape[0], 1, shape[2], 1))
	return nd.concat(range_x, range_y, dim = 1)


class PipelineFlownet:
	_lr = None

	def __init__(self, ctx, config):
		self.ctx = ctx
		self.network = build_network(getattr(config.network, 'class').get('MaskFlownet'))(config=config)
		self.network.hybridize()
		self.network.collect_params().initialize(init=mx.initializer.MSRAPrelu(slope=0.1), ctx=self.ctx)
		self.trainer = gluon.Trainer(self.network.collect_params(), 'adam', {'learning_rate': 1e-4})
		self.strides = self.network.strides or [64, 32, 16, 8, 4]

		self.scale = self.strides[-1]
		self.upsampler = Upsample(self.scale)
		self.upsampler_mask = Upsample(self.scale)

		self.epeloss = EpeLoss()
		self.epeloss.hybridize()
		self.epeloss_with_mask = EpeLossWithMask()
		self.epeloss_with_mask.hybridize()

		multiscale_weights = config.network.mw.get([.005, .01, .02, .08, .32])
		if len(multiscale_weights) != 5:
			multiscale_weights = [.005, .01, .02, .08, .32]
		self.multiscale_epe = MultiscaleEpe(
			scales = self.strides, weights = multiscale_weights, match = 'upsampling',
			eps = 1e-8, q = config.optimizer.q.get(None))
		self.multiscale_epe.hybridize()

		self.reconstruction = Reconstruction2DSmooth(3)
		self.reconstruction.hybridize()

		self.lr_schedule = config.optimizer.learning_rate.value

	def save(self, prefix):
		self.network.save_parameters(prefix + '.params')
		self.trainer.save_states(prefix + '.states')

	def load(self, checkpoint):
		self.network.load_parameters(checkpoint, ctx=self.ctx)

	def load_head(self, checkpoint):
		self.network.load_head(checkpoint, ctx=self.ctx)

	def fix_head(self):
		self.network.fix_head()

	def set_learning_rate(self, steps):
		i = 0
		while i < len(self.lr_schedule) and steps > self.lr_schedule[i][0]:
			i += 1
		try:
			lr = self.lr_schedule[i][1]
		except IndexError:
			return False
		self.trainer.set_learning_rate(lr)
		self._lr = lr
		return True	

	@property
	def lr(self):
		return self._lr

	def loss(self, pred, occ_masks, labels, masks):
		loss = self.multiscale_epe(labels, masks, *pred)
		return loss
	
	def centralize(self, img1, img2):
		rgb_mean = nd.concat(img1, img2, dim = 2).mean(axis = (2, 3)).reshape((-2, 1, 1))
		return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

	def train_batch(self, img1, img2, label, geo_aug, color_aug, mask = None):
		losses = []
		epes = []
		batch_size = img1.shape[0]
		if mask is None:
			mask = np.full(shape = (batch_size, 1, 1, 1), fill_value = 255, dtype = np.uint8)
		img1, img2, label, mask = map(lambda x : gluon.utils.split_and_load(x, self.ctx), (img1, img2, label, mask))
		
		with autograd.record():
			for img1s, img2s, labels, masks in zip(img1, img2, label, mask):
				img1s, img2s, labels, masks = img1s / 255.0, img2s / 255.0, labels.astype("float32", copy=False), masks / 255.0
				img1s, img2s, labels, masks = geo_aug(img1s, img2s, labels, masks)
				img1s, img2s = color_aug(img1s, img2s)
				img1s, img2s, rgb_mean = self.centralize(img1s, img2s)
				pred, occ_masks, warpeds = self.network(img1s, img2s)
				
				labels = labels.flip(axis = 1)
				loss = self.loss(pred, occ_masks, labels, masks)
				epe = self.epeloss_with_mask(self.upsampler(pred[-1]), labels, masks)

				losses.append(loss)
				epes.append(epe)
		
		for loss in losses:
			loss.backward()
		self.trainer.step(batch_size)
		return {"epe": np.mean(np.concatenate([epe.asnumpy() for epe in epes]))}

	def do_batch_mx(self, img1, img2, resize = None):
		''' do a batch of samples range [0,1] with network preprocessing and padding
		'''
		img1, img2, _ = self.centralize(img1, img2)
		shape = img1.shape
		if resize is None:
			pad_h = (64 - shape[2] % 64) % 64
			pad_w = (64 - shape[3] % 64) % 64
		else:
			pad_h = resize[0] - shape[2]
			pad_w = resize[1] - shape[3]
		if pad_h != 0 or pad_w != 0:
			img1 = nd.contrib.BilinearResize2D(img1, height = shape[2] + pad_h, width = shape[3] + pad_w)
			img2 = nd.contrib.BilinearResize2D(img2, height = shape[2] + pad_h, width = shape[3] + pad_w)
		pred, flows, warpeds = self.network(img1, img2)
		return pred, flows, warpeds

	def do_batch(self, img1, img2, label = None, mask = None, resize = None):
		shape = img1.shape
		flows, occ_masks, _ = self.do_batch_mx(img1, img2, resize = resize)
		flow = self.upsampler(flows[-1])
		occ_mask = self.upsampler(occ_masks[0])
		if shape[2] != flow.shape[2] or shape[3] != flow.shape[3]:
			flow = nd.contrib.BilinearResize2D(flow, height = shape[2], width = shape[3]) * nd.array(
				[shape[d] / flow.shape[d] for d in (2, 3)], ctx = flow.context).reshape((1, 2, 1, 1))
			occ_mask = nd.contrib.BilinearResize2D(occ_mask, height = shape[2], width = shape[3])
		warp = self.reconstruction(img2, flow)
		epe = None
		if label is not None and mask is not None:
			epe = self.epeloss_with_mask(flow, label, mask)
		return flow, occ_mask, warp, epe

	def validate(self, img1, img2, label, mask = None, batch_size = 1, resize = None, return_type = 'epe'):
		''' validate the whole dataset
		'''
		np_epes = []
		size = len(img1)
		bs = batch_size
		if mask is None:
			mask = [np.full(shape = (1, 1, 1), fill_value = 255, dtype = np.uint8)] * size
		for j in range(0, size, bs):
			batch_img1 = img1[j: j + bs]
			batch_img2 = img2[j: j + bs]
			batch_label = label[j: j + bs]
			batch_mask = mask[j: j + bs]

			batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
			batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))
			batch_label = np.transpose(np.stack(batch_label, axis=0), (0, 3, 1, 2))
			batch_mask = np.transpose(np.stack(batch_mask, axis=0), (0, 3, 1, 2))

			def Norm(x):
				return nd.sqrt(nd.sum(nd.square(x), axis = 1, keepdims = True))

			batch_epe = []
			ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
			nd_img1, nd_img2, nd_label, nd_mask = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), (batch_img1, batch_img2, batch_label, batch_mask))
			for img1s, img2s, labels, masks in zip(nd_img1, nd_img2, nd_label, nd_mask):
				img1s, img2s, labels, masks = img1s / 255.0, img2s / 255.0, labels.astype("float32", copy=False), masks / 255.0
				labels = labels.flip(axis = 1)
				flows, _, _, epe = self.do_batch(img1s, img2s, labels, masks, resize = resize)

				# calculate the metric for kitti dataset evaluation
				if return_type is not 'epe':
					eps = 1e-8
					epe = ((Norm(flows - labels) > 3) * ((Norm(flows - labels) / (Norm(labels) + eps)) > 0.05) * masks).sum(axis=0, exclude=True) / masks.sum(axis=0, exclude=True)
				
				batch_epe.append(epe)
			np_epes.append(np.concatenate([epe.asnumpy() for epe in batch_epe]))
		
		return np.mean(np.concatenate(np_epes, axis = 0), axis = 0)

	def predict(self, img1, img2, batch_size, resize = None):
		''' predict the whole dataset
		'''
		size = len(img1)
		bs = batch_size
		for j in range(0, size, bs):
			batch_img1 = img1[j: j + bs]
			batch_img2 = img2[j: j + bs]

			batch_img1 = np.transpose(np.stack(batch_img1, axis=0), (0, 3, 1, 2))
			batch_img2 = np.transpose(np.stack(batch_img2, axis=0), (0, 3, 1, 2))

			batch_flow = []
			batch_occ_mask = []
			batch_warped = []

			ctx = self.ctx[ : min(len(batch_img1), len(self.ctx))]
			nd_img1, nd_img2 = map(lambda x : gluon.utils.split_and_load(x, ctx, even_split = False), (batch_img1, batch_img2))
			for img1s, img2s in zip(nd_img1, nd_img2):
				img1s, img2s = img1s / 255.0, img2s / 255.0
				flow, occ_mask, warped, _ = self.do_batch(img1s, img2s, resize = resize)
				batch_flow.append(flow)
				batch_occ_mask.append(occ_mask)
				batch_warped.append(warped)
			
			flow = np.concatenate([x.asnumpy() for x in batch_flow])
			occ_mask = np.concatenate([x.asnumpy() for x in batch_occ_mask])
			warped = np.concatenate([x.asnumpy() for x in batch_warped])
			
			flow = np.transpose(flow, (0, 2, 3, 1))
			flow = np.flip(flow, axis = -1)
			occ_mask = np.transpose(occ_mask, (0, 2, 3, 1))
			warped = np.transpose(warped, (0, 2, 3, 1))
			for k in range(len(flow)):
				yield flow[k], occ_mask[k], warped[k]
