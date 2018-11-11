import numpy as np 
import pickle

# =================================================================================
# build data
# =================================================================================
def shuffle(x, y):
	i = np.random.permutation(len(y))
	return x[i], y[i]

def data_normalize(x):
	return x / 255 * 2 - 1

def load_data(path):
	x, y = pickle.load(open(path, 'rb'))
	return x, y

# =================================================================================
# bounding box related
# =================================================================================
def xy_to_cwh(box_xy):
	# intput: x1, y1, x2, y2
	# output: xc, yc, w, h
	x1, y1, x2, y2 = box_xy
	xc = (x1 + x2) / 2
	yc = (y1 + y2) / 2
	w = x2 - x1
	h = y2 - y1
	return [xc, yc, w, h]

def box_xy_resize(h_ori, w_ori, h_resize, w_resize, box_xy):
	w_ratio = 1. * w_resize / w_ori
	h_ratio = 1. * h_resize / h_ori

	x1, y1, x2, y2 = box_xy
	x1_resize = x1 * w_ratio
	x2_resize = x2 * w_ratio
	y1_resize = y1 * h_ratio
	y2_resize = y2 * h_ratio

	return [x1_resize, y1_resize, x2_resize, y2_resize]

def normalize_box_cwh(box_cwh, h, w, grid_num):
	x, y, bw, bh = box_cwh
	w_norm = 1. * bw / w
	h_norm = 1. * bh / h 

	w_grid = 1. * w / grid_num
	h_grid = 1. * h / grid_num
	col = int(x / w_grid)
	row = int(y / h_grid)

	x_norm = 1. * (x - col*w_grid) / w_grid
	y_norm = 1. * (y - row*h_grid) / h_grid
	cwh_norm = [x_norm, y_norm, w_norm, h_norm]
	grid = [row, col]
	return cwh_norm, grid


