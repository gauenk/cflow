
# -- python import --
import glob
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from einops import rearrange
from pathlib import Path

# -- swig import --
import sys
sys.path.append("./pylib/")
import oflow

# -- flow viz --
from flow_vis import flow_to_color

def save_flow_image(path,flow_uv):
    flow_color = flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False)
    img = Image.fromarray(flow_color)
    img.save(path)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = np.asarray(image)
    image = image[:,:,::-1] # BGR
    return image

def load_mat(path):
    flow_mat = loadmat(path)
    vx = flow_mat['vx']
    vy = flow_mat['vy']
    flow = np.r_[vx[None,:],vy[None,:]]
    flow = rearrange(flow,'two h w -> h w two')
    return flow

def load_file_paths(path):
    glob_path = str(path)+"/*"
    files = []
    for filename in glob.glob(glob_path):
        stem = Path(filename).stem
        num = int(stem[-2:])
        files.append((num,filename))
    ordered_files = [None for _ in range(len(files))]
    for (num,fn) in files:
        ordered_files[num-1] = fn
    return ordered_files

def load_frame_and_flows(path):

    # -- get paths to images --
    frame_path = path / Path("./frames")
    frame_paths = load_file_paths(frame_path)

    flow_path = path / Path("./flows")
    flow_paths = load_file_paths(flow_path)

    # -- load image frames --
    frames = []
    for frame_fn in frame_paths:
        frame = load_image(frame_fn)
        frames.append(frame)

    # -- load image flows --
    flows = []
    for flow_fn in flow_paths:
        flow = load_mat(flow_fn)
        flows.append(flow)

    return frames,flows

# -- run tests --

compute_optical_flow = oflow.OpticalFlow_ComputeOpticalFlow
test_numpy_fxn = oflow.OpticalFlow_test_numpy_call
table_path = Path("./data/table/")
frames,flows = load_frame_and_flows(table_path)
nframes = len(frames)
save_dir = Path("./output")
if not save_dir.exists(): save_dir.mkdir()

for t in range(nframes-1):

    im1 = frames[t].astype(np.double)/255.
    im2 = frames[t+1].astype(np.double)/255.

    flow_gt = flows[t].astype(np.double)
    flow_est = np.zeros(flow_gt.shape).astype(np.double)

    h,w,c = im1.shape
    im1_swig = oflow.swig_ptr(im1)
    im2_swig = oflow.swig_ptr(im2)
    flow_est_swig = oflow.swig_ptr(flow_est)
    compute_optical_flow(im1_swig,im2_swig,flow_est_swig,h,w,c)

    # -- visually compare output --
    # print(np.concatenate([flow_est,flow_gt],axis=-1))

    flow_fn = save_dir / Path(f"./flow_frame_est_{t}.png")
    save_flow_image(flow_fn,flow_est)

    flow_fn = save_dir / Path(f"./flow_frame_gt_{t}.png")
    save_flow_image(flow_fn,flow_gt)

    error = np.mean(np.abs(flow_est - flow_gt))
    print("Error: %2.2f" % error)


