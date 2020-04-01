import os
from reader import sintel, kitti
import cv2
import numpy as np

# PLEASE MODIFY the paths specified in sintel.py and kitti.py

def predict(pipe, prefix, batch_size = 8, resize = None):

	sintel_resize = (448, 1024) if resize is None else resize
	sintel_dataset = sintel.list_data()
	prefix = prefix + '_sintel'
	if not os.path.exists(prefix):
		os.mkdir(prefix)
	
	flo = sintel.Flo(1024, 436)

	for div in ('test',):
		for k, dataset in sintel_dataset[div].items():
			if k == 'clean':
				continue
			output_folder = os.path.join(prefix, k)
			if not os.path.exists(output_folder):
				os.mkdir(output_folder)
			img1, img2 = [[sintel.load(p) for p in data] for data in list(zip(*dataset))[:2]]
			for result, entry in zip(pipe.predict(img1, img2, batch_size = 1, resize = sintel_resize), dataset):
				flow, occ_mask, warped = result
				img1 = entry[0]
				fname = os.path.basename(img1)
				seq = os.path.basename(os.path.dirname(img1))
				seq_output_folder = os.path.join(output_folder, seq)
				if not os.path.exists(seq_output_folder):
					os.mkdir(seq_output_folder)
				flo.save(flow, os.path.join(seq_output_folder, fname.replace('.png', '.flo')))

	'''
	KITTI 2012: 
	Submission instructions: For the optical flow benchmark, all flow fields of the test set must be provided in the root directory of a zip file using the file format described in the readme.txt (16 bit color png) and the file name convention of the ground truth (000000_10.png, ... , 000194_10.png).

	KITTI 2015:
	Submission instructions: Provide a zip file which contains the 'disp_0' directory (stereo), the 'flow' directory (flow), or the 'disp_0', 'disp_1' and 'flow' directories (scene flow) in its root folder. Use the file format and naming described in the readme.txt (000000_10.png,...,000199_10.png). 
	'''

	kitti_resize = (512, 1152) if resize is None else resize
	kitti_dataset = kitti.read_dataset_testing(resize = kitti_resize)
	prefix = prefix.replace('sintel', 'kitti')
	if not os.path.exists(prefix):
		os.mkdir(prefix)

	for k, dataset in kitti_dataset.items():
		output_folder = os.path.join(prefix, k)
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)

		img1 = kitti_dataset[k]['image_0']
		img2 = kitti_dataset[k]['image_1']
		cnt = 0
		for result in pipe.predict(img1, img2, batch_size = 1, resize = kitti_resize):
			flow, occ_mask, warped = result
			out_name = os.path.join(output_folder, '%06d_10.png' % cnt)
			cnt = cnt + 1

			pred = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.uint16)
			pred[:, :, 2] = (64.0 * (flow[:, :, 0] + 512)).astype(np.uint16)
			pred[:, :, 1] = (64.0 * (flow[:, :, 1] + 512)).astype(np.uint16)
			cv2.imwrite(out_name, pred)
			