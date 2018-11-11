import argparse
import sys
import config
import utils

from lenet5 import Lenet5

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Lenet5', help=' | '.join(config.model_names))
parser.add_argument('--mode', default='train', help='train | predict')



if __name__ == '__main__':
	args = parser.parse_args()
	model_name, mode = args.model, args.mode
	if model_name not in config.model_names:
		print('Invalid model! Please choose from: ', *config.model_names)
		sys.exit()

	data_dir = config.data_dir[model_name]
	x_train, y_train = utils.load_data(data_dir + 'train.p')
	x_dev, y_dev     = utils.load_data(data_dir + 'dev.p')
	x_test, y_test   = utils.load_data(data_dir + 'test.p')

	if model_name == 'Lenet5':
		model = Lenet5(x_train, y_train, x_dev, y_dev, x_test, y_test)


	if mode == 'train':
		model.train()