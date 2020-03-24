import os

# ======== PLEASE MODIFY ========
things3d_root = r'path\to\your\FlyingThings3D_subset'

def list_data(path = None, sub_type = 'clean'):
	if path is None:
		path = things3d_root
	parts = ('train', 'val')
	sub_types = ('sub_type',)
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
					flow_ind = int(orient == 'into_past')
					path_image = os.path.join(path, part, 'image_' + sub_type, camera)
					path_flow = os.path.join(path, part, 'flow', camera, orient)
					num_files = len(os.listdir(path_flow)) - 1
					for k in range(num_files):
						dataset['image_0'].append(os.path.join(path_image, '%07d.png' % (k + flow_ind)))
						dataset['image_1'].append(os.path.join(path_image, '%07d.png' % (k + 1 - flow_ind)))
						dataset['flow'].append(os.path.join(path_flow, '%07d.flo' % (k + flow_ind)))

	return dataset

if __name__ == '__main__':
    dataset = list_data()
    print(len(dataset['flow']))
    print(dataset['flow'][-1])