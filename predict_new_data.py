import os
import sys

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = r'.'
# to CUDA\vX.Y\bin
#os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']

import argparse
import yaml
import numpy as np
import mxnet as mx
import cv2
import flow_vis
from moviepy.editor import ImageSequenceClip
from moviepy.audio.AudioClip import AudioArrayClip

import network.config
from network import get_pipeline
import path
import logger


def find_checkpoint(checkpoint_str):
    # find checkpoint
    steps = 0
    if checkpoint_str is not None:
    	if ':' in checkpoint_str:
    		prefix, steps = checkpoint_str.split(':')
    	else:
    		prefix = checkpoint_str
    		steps = None
    	log_file, run_id = path.find_log(prefix)	
    	if steps is None:
    		checkpoint, steps = path.find_checkpoints(run_id)[-1]
    	else:
    		checkpoints = path.find_checkpoints(run_id)
    		try:
    			checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
    		except StopIteration:
    			print('The steps not found in checkpoints', steps, checkpoints)
    			sys.stdout.flush()
    			raise StopIteration
    	steps = int(steps)
    	if args.clear_steps:
    		steps = 0
    	else:
    		_, exp_info = path.read_log(log_file)
    		exp_info = exp_info[-1]
    		for k in args.__dict__:
    			if k in exp_info and k in ('tag',):
    				setattr(args, k, eval(exp_info[k]))
    				print('{}={}, '.format(k, exp_info[k]), end='')
    		print()
    	sys.stdout.flush()
    return checkpoint, steps


def load_model(config_str):
    # load network configuration
    with open(os.path.join(repoRoot, 'network', 'config', config_str)) as f:
    	config =  network.config.Reader(yaml.load(f))
    return config


def instantiate_model(gpu_device, config):
    ctx = [mx.cpu()] if gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, gpu_device.split(','))]
    # initiate
    pipe = get_pipeline(args.network, ctx=ctx, config=config)
    return pipe
    

def load_checkpoint(pipe, config, checkpoint):
    # load parameters from given checkpoint
    print('Load Checkpoint {}'.format(checkpoint))
    sys.stdout.flush()
    network_class = getattr(config.network, 'class').get()
    print('load the weight for the network')
    pipe.load(checkpoint)
    if network_class == 'MaskFlownet':
   		print('fix the weight for the head network')
   		pipe.fix_head()
    sys.stdout.flush()
    return pipe


def predict_image_pair_flow(img1, img2, pipe, resize=None):
    for result in pipe.predict([img1], [img2], batch_size = 1, resize=resize):
        flow, occ_mask, warped = result
    return flow, occ_mask, warped
    

def create_video_clip_from_frames(frame_list, fps):
    """ Function takes a list of video frames and puts them together in a sequence"""
    visual_clip = ImageSequenceClip(frame_list, fps=fps) #put frames together using moviepy
    return visual_clip #return the ImageSequenceClip


def predict_video_flow(video_filename, batch_size, resize=None):
    cap = cv2.VideoCapture(video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_frames = []
    new_frames = []
    has_frames, frame = cap.read()
    prev_frames.append(frame)
    while True:
        has_frames, frame = cap.read()
        if not has_frames:
            cap.release()
            break
        new_frames.append(frame)
        prev_frames.append(frame)
    del prev_frames[-1] #delete the last frame of the video from prev_frames
    flow_video = [flow for flow, occ_mask, warped in pipe.predict(prev_frames, new_frames, batch_size=batch_size, resize=resize)]
    
    return flow_video, fps



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('flow_filepath', type=str, help='destination filepath of the flow image/video')
    parser.add_argument('config', type=str, nargs='?', default=None)
    parser.add_argument('--image_1', type=str, help='filepath of the first image')
    parser.add_argument('--image_2', type=str, help='filepath of the second image')
    parser.add_argument('--video_filepath', type=str, help='filepath of the input video')
    parser.add_argument('-g', '--gpu_device', type=str, default='', help='Specify gpu device(s)')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, 
    	help='model checkpoint to load; by default, the latest one.'
    	'You can use checkpoint:steps to load to a specific steps')
    parser.add_argument('--clear_steps', action='store_true')
    parser.add_argument('-n', '--network', type=str, default='MaskFlownet', help='The choice of network')
    parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')
    parser.add_argument('--resize', type=str, default='', help='shape to resize image frames before inference')
    parser.add_argument('--threads', type=str, default=8, help='Number of threads to use when writing flow video to file')
    
    args = parser.parse_args()
    
    
    # Get desired image resize from the string argument
    infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None
    
    checkpoint, steps = find_checkpoint(args.checkpoint)
    config = load_model(args.config)
    pipe = instantiate_model(args.gpu_device, config)
    pipe = load_checkpoint(pipe, config, checkpoint)
    
    if args.image_1 is not None:
        image_1 = cv2.imread(args.image_1)
        image_2 = cv2.imread(args.image_2)
        flow, occ_mask, warped = predict_image_pair_flow(image_1, image_2, pipe)
        cv2.imwrite(args.flow_filepath, flow_vis.flow_to_color(flow, convert_to_bgr=False))
    else:
        flow_video, fps = predict_video_flow(args.video_filepath, batch_size=args.batch)
        flow_video_visualisations = [flow_vis.flow_to_color(flow, convert_to_bgr=False) for flow in flow_video]
        flow_video_clip = create_video_clip_from_frames(flow_video_visualisations, fps)
        flow_video_clip.write_videofile(args.flow_filepath, threads=args.threads, logger=None) #export the video

    sys.exit(0)
