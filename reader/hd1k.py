import os
import cv2
import numpy as np

VALIDATE_INDICES = [5]

# ======== PLEASE MODIFY ========
hd1k_root = r'path\to\your\HD1K'

hd1k_image = os.path.join(hd1k_root, r'hd1k_input\image_2')
hd1k_flow_occ = os.path.join(hd1k_root, r'hd1k_flow_gt\flow_occ')
hd1k_path = dict()
hd1k_path['image'] = hd1k_image
hd1k_path['flow_occ'] = hd1k_flow_occ

def read_dataset(path = None, parts = 'mixed', resize = None, samples = -1, normalize = True, crop = (50, 100)):
	if path is None:
		path = hd1k_path
	dataset = dict()
	dataset['image_0'] = []
	dataset['image_1'] = []
	dataset['flow'] = []
	dataset['occ'] = []
	path_images = path['image']
	path_flows = path['flow_occ']
	list_files = os.listdir(path_flows)
	num_files = len(list_files) - 1
	ind_valids = VALIDATE_INDICES
	num_valids = len(ind_valids)
	if samples is not -1:
		num_files = min(num_files, samples)
	ind = 0
	i_pre = -1
	i_cur = 0
	j_cur = 0
	for k in range(num_files):
		if ind < num_valids and ind_valids[ind] == k:
			ind += 1
			if parts == 'train':
				continue
		elif parts == 'valid':
			continue
		i_cur = (int) (list_files[k][-15:-9])
		j_cur = (int) (list_files[k][-8:-4])
		flag = False
		if i_cur != i_pre:
			flag = True
		i_pre = i_cur
		if flag:
			continue
		img0 = cv2.imread(os.path.join(path_images, '%06d_%04d.png' % (i_cur, j_cur - 1)))[crop[0]: -crop[0], crop[1]: -crop[1]]
		img1 = cv2.imread(os.path.join(path_images, '%06d_%04d.png' % (i_cur, j_cur)))[crop[0]: -crop[0], crop[1]: -crop[1]]
		flow_occ = cv2.imread(os.path.join(path_flows, '%06d_%04d.png' % (i_cur, j_cur - 1)), -1)[crop[0]: -crop[0], crop[1]: -crop[1]]
		if normalize:
			img_min, img_max = min(img0.min(), img1.min()), max(img0.max(), img1.max())
			img0, img1 = [((img - img_min) * (255.0 / (img_max - img_min))).astype(np.uint8) for img in (img0, img1)]
		flow = np.flip(flow_occ[..., 1:3], axis=-1).astype(np.float32)
		flow = (flow - 32768.) / (64.)
		occ = flow_occ[..., 0:1].astype(np.uint8)
		flow = flow * occ
		# flow_avg = np.zeros((flow.shape[0] + 2, flow.shape[1] + 2, flow.shape[2]))
		# occ_avg = np.zeros((occ.shape[0] + 2, occ.shape[1] + 2, occ.shape[2]))
		# for i in range(3):
		#	 for j in range(3):
		#		 flow_avg += np.pad(flow, [(i, 2 - i), (j, 2 - j), (0, 0)], 'constant')
		#		 occ_avg += np.pad(occ, [(i, 2 - i), (j, 2 - j), (0, 0)], 'constant')
		# occ_avg[occ_avg == 0] = 10 
		# flow += flow_avg[1:-1, 1:-1, ...] / occ_avg[1:-1, 1:-1, ...] * (1 - occ)
		if resize is not None:
			img0 = cv2.resize(img0, resize)
			img1 = cv2.resize(img1, resize)
			flow = cv2.resize(flow, resize) * ((np.array(resize, dtype = np.float32) - 1.0) / (
				np.array([flow.shape[d] for d in (1, 0)], dtype = np.float32) - 1.0))[np.newaxis, np.newaxis, :]
			occ = cv2.resize(occ.astype(np.float32), resize)[..., np.newaxis]
			flow = flow / (occ + (occ == 0))
			occ = (occ * 255).astype(np.uint8)
		else:
			occ = occ * 255
		dataset['image_0'].append(img0)
		dataset['image_1'].append(img1)
		dataset['flow'].append(flow)
		dataset['occ'].append(occ)
	return dataset


if __name__ == '__main__':
	dataset = read_dataset()

