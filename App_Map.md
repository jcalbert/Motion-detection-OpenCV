"""
Created on Thu Jun 21 21:11:53 2018

@author: jcalbert
"""

General approach to an app:

1) Read in a file (by name)

2) Process video to quantify motion
 2.1) Display progress, somehow
 
3) Read out motion time series, allow user to adjust parameters for 
clip bounds.

4) Present preview of all clips
 a) Action Sequence (composite)
 b) Side-by-side stills
 c) animated thumbnail
 
5) Select clips, use bounds to extract video segments (e.g. by ffmpeg, mencoder))
 5.1) Optional - Stitch all clips together in one edit.