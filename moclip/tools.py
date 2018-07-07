"""
Tools, especially for working with static video files.
"""
import cv2
import numpy as np

import time

from scipy.ndimage import maximum_filter1d

import os
from subprocess import Popen

def get_motion_series(detector, frame_skip = None, preview=False):
    """
    Given an initialzed motion detector, produce a time series
    of the esitmated motion level for each frame.
    """
    n_frames = int(detector.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_skip is None:
        frame_ids = np.arange(0,n_frames)
    else:
        raise NotImplementedError("Frame skip not yet implemented")
        frame_ids = np.arange(0, n_frames, frame_skip)
        
    motion_series = np.zeros(n_frames)
    if preview:
        cv2.namedWindow("Preview")
        dt = [0,time.time()]

    for i in frame_ids[1:]:
        detector.step()
        motion_series[i] = detector.quant.motion_level
        
        if preview:
            dt[1] = time.time()
            if dt[1]-dt[0] > 1.0:
                dt[0] = dt[1]
                cv2.imshow("Preview",detector.visualize())
                _=cv2.waitKey(1) % 0x100

    return motion_series

def get_clips(series, threshold, warmup = 30, cooldown = 30):    
    """
    Given a time series, a threshold, detect preiods of motion and return
    a list of (start_frame, end_frame) tuples.
    
    A motion period is padded by `warmup` frames before and `cooldown` frames
    after with motion levels below `threshold`.
    """
    maxed_series = maximum_filter1d(series,
                                    warmup+cooldown+1,
                                    origin=(-warmup + cooldown)/2)
    moving_frames = (maxed_series > threshold).astype('float')
    moving_frames[0] = 0
    moving_frames[-1] = 0
    on_off = np.diff(moving_frames)
    starts = np.where(on_off == 1)[0]
    stops  = np.where(on_off == -1)[0]

    return zip(starts, stops)

def split_vid(fpath, clip_bounds):
    """ 
    This is really a convenience script to invoke a video encoder 
    (e.g. ffmpeg, mencoder) to extract segments from a video.  Currently
    not tool-agnositc.
    
    fpath - path to the original file.
    
    clip_bounds - list of (start, end) tuples to make clips out of.
    """

    cap = cv2.VideoCapture(fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    del(cap)
    
    fdir, fname = os.path.split(fpath)
    fbase, fext = os.path.splitext(fname)
    
    dst = os.path.join(fdir, fbase + '_CLIPS')

    if os.path.exists(dst):
        pass
    else:
        os.mkdir(dst)
    
    cmd_base = ['mencoder',
                fpath,
                '-oac','pcm',
                '-ovc','copy']

    if '.mts' in fpath.lower():
        cmd_base.append('-demuxer')
        cmd_base.append('lavf')
        #cmd_base.append('-of')
        #cmd_base.append('lavf=mp4')
    
    
    for i, cb in enumerate(clip_bounds):

        this_file = 'clip_' + str(i).zfill(5) +'.' + fext
        this_out = os.path.join(dst, this_file)
        
        start_time = round(cb[0] / fps, 2)
        end_time = round( (cb[1]-cb[0]) / fps, 2)
        
        this_cmd = ['-ss', str(start_time),
                    '-endpos', str(end_time),
                    '-o', this_out]
        print "Writing: {}".format(this_out)
        proc = Popen(cmd_base + this_cmd)
        proc.wait()


from imutils.video import FileVideoStream

class FVS_CAP(FileVideoStream):
    """
    This extends imutils' FileVideoStream to a drop-in replacement for a
    cv2 VideoCapture object, allowing one to read capture properties like
    frame rate via self.get(property)
    """
    def __init__(self, name, queueSize=128):
        FileVideoStream.__init__(self, name, queueSize=queueSize)

    def get(self, prop):
        return self.stream.get(prop)
    
    def read(self):
        return (self.more(), FileVideoStream.read(self))