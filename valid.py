from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import os
from reader import sintel, kitti
import skimage.io
import cv2
import numpy as np


def make_colorwheel():
	'''
	Generates a color wheel for optical flow visualization as presented in:
		Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
		URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

	According to the C++ source code of Daniel Scharstein
	According to the Matlab source code of Deqing Sun
	'''

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR
	colorwheel = np.zeros((ncols, 3))
	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
	col = col+RY
	# YG
	colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
	colorwheel[col:col+YG, 1] = 255
	col = col+YG
	# GC
	colorwheel[col:col+GC, 1] = 255
	colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
	col = col+GC
	# CB
	colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
	colorwheel[col:col+CB, 2] = 255
	col = col+CB
	# BM
	colorwheel[col:col+BM, 2] = 255
	colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
	col = col+BM
	# MR
	colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
	colorwheel[col:col+MR, 0] = 255
	return colorwheel


def flow_compute_color(u, v, convert_to_bgr=False):
	'''
	Applies the flow color wheel to (possibly clipped) flow components u and v.

	According to the C++ source code of Daniel Scharstein
	According to the Matlab source code of Deqing Sun

	:param u: np.ndarray, input horizontal flow
	:param v: np.ndarray, input vertical flow
	:param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
	:return:
	'''

	flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

	colorwheel = make_colorwheel()  # shape [55x3]
	ncols = colorwheel.shape[0]

	rad = np.sqrt(np.square(u) + np.square(v))
	a = np.arctan2(-v, -u)/np.pi

	fk = (a+1) / 2*(ncols-1)
	k0 = np.floor(fk).astype(np.int32)
	k1 = k0 + 1
	k1[k1 == ncols] = 0
	f = fk - k0

	for i in range(colorwheel.shape[1]):

		tmp = colorwheel[:,i]
		col0 = tmp[k0] / 255.0
		col1 = tmp[k1] / 255.0
		col = (1-f)*col0 + f*col1

		idx = (rad <= 1)
		col[idx]  = 1 - rad[idx] * (1-col[idx])
		col[~idx] = col[~idx] * 0.75   # out of range?

		# Note the 2-i => BGR instead of RGB
		ch_idx = 2-i if convert_to_bgr else i
		flow_image[:,:,ch_idx] = np.floor(255 * col)

	return flow_image


def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
	'''
	Expects a two dimensional flow image of shape [H,W,2]

	According to the C++ source code of Daniel Scharstein
	According to the Matlab source code of Deqing Sun

	:param flow_uv: np.ndarray of shape [H,W,2]
	:param clip_flow: float, maximum clipping value for flow
	:return:
	'''

	assert flow_uv.ndim == 3, 'input flow must have three dimensions'
	assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

	#if clip_flow is not None:
	#	flow_uv = np.clip(flow_uv, 0, clip_flow)

	u = flow_uv[:,:,0]
	v = flow_uv[:,:,1]

	rad = np.sqrt(np.square(u) + np.square(v))
	rad_max = np.max(rad)
	
	if clip_flow is not None:
		rad_max = min(rad_max, clip_flow)

	epsilon = 1e-5
	u = u / (rad_max + epsilon)
	v = v / (rad_max + epsilon)
	
	if clip_flow is not None:
		u = u / np.maximum(rad / rad_max, 1)
		v = v / np.maximum(rad / rad_max, 1)

	return flow_compute_color(u, v, convert_to_bgr)


def predict(pipe, prefix, batch_size = 8, resize = None):

	sintel_resize = (448, 1024) if resize is None else resize
	sintel_dataset = sintel.list_data(sintel.sintel_path)
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
				skimage.io.imsave(os.path.join(seq_output_folder, fname), flow_to_color(flow, clip_flow = 20))

	'''
	KITTI 2012: 
	Submission instructions: For the optical flow benchmark, all flow fields of the test set must be provided in the root directory of a zip file using the file format described in the readme.txt (16 bit color png) and the file name convention of the ground truth (000000_10.png, ... , 000194_10.png).

	KITTI 2015:
	Submission instructions: Provide a zip file which contains the 'disp_0' directory (stereo), the 'flow' directory (flow), or the 'disp_0', 'disp_1' and 'flow' directories (scene flow) in its root folder. Use the file format and naming described in the readme.txt (000000_10.png,...,000199_10.png). 
	'''

	kitti_resize = (512, 1152) if resize is None else resize
	kitti_dataset = kitti.read_dataset_testing(resize = kitti_resize)
	prefix = prefix + '_kitti'
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
			skimage.io.imsave(out_name.replace('.png', '_flow.png'), flow_to_color(flow, clip_flow = 20))
			