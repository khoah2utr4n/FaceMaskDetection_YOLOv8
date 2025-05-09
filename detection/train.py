import argparse
from model import trainModel, getModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model yolov8 for Face Mask Detection.')
    parser.add_argument('--epochs', default=50, type=int,
                        help='number of epochs for training')
    parser.add_argument('--resume', default=False, type=bool,
                        help='continue training from the last epoch or not')
    parser.add_argument('--weightsPath', default='yolov8n.pt', type=str,
                        help="Path to the model's weights (default: 'yolov8n.pt')")

    args = parser.parse_args()
    model = getModel(args.weightsPath)
    trainingResults = trainModel(model, args.epochs, resume=args.resume)