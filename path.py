import os
import re
import sys

repo_root = os.path.dirname(__file__)

def list_dir(folder, pattern, groups=False):
	pattern = re.compile(pattern)
	for f in os.listdir(folder):
		m = pattern.match(f)
		if m is not None:
			if groups:
				yield (os.path.join(folder, f),) + m.groups()
			else:
				yield os.path.join(folder, f)

def find_log(prefix):
	log_dir = os.path.join(repo_root, 'logs')
	pattern = r'^(%s(.*\d)?)\.log$' % prefix
	rets = list(list_dir(log_dir, pattern, groups=True))
	print(rets)
	if len(rets) > 0 :
		return rets[0][:2]
	else:
		raise ValueError("Not found {}".format(prefix))

def find_checkpoints(run_id):
	weights_dir = os.path.join(repo_root, 'weights')
	pattern = r'^{}.*_(\d+)\.params$'.format(run_id)
	checkpoints = list(sorted(list_dir(weights_dir, pattern, groups=True), key=lambda x : int(x[1])))
	return checkpoints

def read_log(fname):
	val = []
	exp_info = []
	in_start = False
	with open(fname, 'r') as fi:
		for ln in fi:
			p = ln.find('] ')
			items = ln[p + 2:].strip().split(', ')
			try:
				kvs = dict([ item.split('=') for item in items ])
			except:
				pass
			if 'val_epe' in kvs:
				val.append(kvs)
			elif 'start' in kvs:
				exp_info.append(kvs)
				in_start = True
			elif in_start:
				exp_info[-1].update(kvs)
				in_start = False
	return val, exp_info
