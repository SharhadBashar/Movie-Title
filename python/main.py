import sys

from train import Train
from predict import Predict

if __name__ == '__main__':
	sys.argv[1] = sys.argv[1].lower() == 'true'
	if (sys.argv[1]) == True:
		print('Starting Training')
		Train()
		print('Finished Training')
	print('Starting Predictions')
	Predict()
	print('Predictions complete')
