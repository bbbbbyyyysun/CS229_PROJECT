import cv2
import numpy as np 
import csv
import pickle
import os

import config
import utils

from tqdm import trange

class_num = config.class_num
grid_num = config.grid_num

def gtsdb(root=config.GTSDB):
	conflict = 0

	labels = np.loadtxt(root+'/gt.txt', delimiter=';', dtype=str)
	image_names = labels[:, 0]
	bounding_box = labels[:, 1:5].astype(float)
	classes = labels[:, 5].astype(int)

	X, Y = [], []
	files = [f for f in os.listdir(root) if f.endswith('.ppm')]
	
	x_size = (448, 448)
	for i in trange(len(files)):
		img = cv2.imread(root + '/' + image_names[i])
		img_resize = cv2.resize(img, x_size)
		X.append(img_resize)

		y = np.zeros((grid_num, grid_num, 5+class_num))
		h_ori, w_ori = img.shape[0:2]
		h_resize, w_resize = img_resize.shape[0:2]
		indices = np.where(image_names == image_names[i])[0]
		
		for index in indices:
			box_xy = bounding_box[index]
			box_xy_resize = utils.box_xy_resize(h_ori, w_ori, h_resize, w_resize, box_xy)
			box_cwh = utils.xy_to_cwh(box_xy_resize)
			(xc, yc, w, h), (row, col) = utils.normalize_box_cwh(box_cwh, h_resize, w_resize, grid_num)

			if y[row, col, 0] == 1:
				conflict += 1
				continue
			
			y[row, col, 0:5] = [1, xc, yc, w, h]
			c = classes[index]
			y[row, col, 5+c] = 1
		Y.append(y)

	X, Y = np.array(X), np.array(Y)
	X, Y = utils.shuffle(X, Y)

	split = len(files) // 10
	x_dev = X[:split]
	y_dev = Y[:split]
	x_test = X[split:2*split]
	y_test = Y[split:2*split]
	x_train = X[2*split:]
	y_train = Y[2*split:]

	print('Build dataset done.')
	print('Train shape:', x_train.shape, y_train.shape)
	print('Val shape:', x_dev.shape, y_dev.shape)
	print('Test shape:', x_test.shape, y_test.shape)
	print('Number of boxes:', bounding_box.shape[0])
	print('Conflict count:', conflict)



def gtsrb(root=config.GTSRB):
	x_train, y_train, x_dev, y_dev, x_test, y_test = [], [], [], [], [], []

	classes = np.arange(0, class_num) # 0-42
	for i in trange(class_num):
		class_name = format(classes[i], '05d')
		prefix = root + '/Images/' + class_name + '/'
		f = open(prefix + 'GT-' + class_name + '.csv')
		reader = csv.reader(f, delimiter=';')
		next(reader, None)

		x, y = [], []
		for row in reader:
			img = cv2.imread(prefix + row[0])
			img = img[np.int(row[4]):np.int(row[6]), np.int(row[3]):np.int(row[5]), :] # np.int()从string转化为int
			# cv2.imshow('img', img)
			# cv2.waitKey(0)
			x.append(img)
			y.append(i)

		x, y = utils.shuffle(np.array(x), np.array(y))
		x, y = x.tolist(), y.tolist()

		split = len(y) // 10
		x_dev += x[:split]
		y_dev += y[:split]
		x_test += x[split:2*split]
		y_test += y[split:2*split]
		x_train += x[2*split:]
		y_train += y[2*split:]
		f.close()

	size = (32, 32)
	x_train = [cv2.resize(x, size) for x in x_train]
	x_dev   = [cv2.resize(x, size) for x in x_dev]
	x_test  = [cv2.resize(x, size) for x in x_test]

	x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train)
	x_dev, y_dev     = np.array(x_dev).astype(np.float32), np.array(y_dev)
	x_test, y_test   = np.array(x_test).astype(np.float32), np.array(y_test)

	x_train, x_dev, x_test = list(map(utils.data_normalize, [x_train, x_dev, x_test]))

	x_train, y_train = utils.shuffle(x_train, y_train)
	x_dev, y_dev     = utils.shuffle(x_dev, y_dev)
	x_test, y_test   = utils.shuffle(x_test, y_test)

	pickle.dump((x_train, y_train), open(root + '/train.p', 'wb'))
	pickle.dump((x_dev, y_dev), open(root + '/dev.p', 'wb'))
	pickle.dump((x_test, y_test), open(root + '/test.p', 'wb'))  # 'w' for write, 'b' for binary; use 'rb' to read
	
	
if __name__ == '__main__':
	#gtsrb()
	gtsdb()

	