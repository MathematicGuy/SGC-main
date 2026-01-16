import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

# python main.py --config=exps/10,10_cifar100.json
# python main.py --config=exps/10,10_imagenet-r.json
# python main.py --config=exps/10,10_imagenet100.json
# python main.py --config=exps/10,10_imagenet1000.json
def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    # parser.add_argument('--config', type=str, default='exps/visual_proto.json',
    parser.add_argument('--config', type=str, default='exps/10,10_cifar100.json',
                        help='Json file of settings.')

	#? Total tasks: 20 task, 10 classes per task
    # parser.add_argument('--config', type=str, default='exps/10,10_imagenet-r.json',
    #                     help='Json file of settings.')

    # parser.add_argument('--config', type=str, default='exps/10,10_imagenet1000.json',
    #                     help='Json file of settings.')

	# parser.add_argument('--config', type=str, default='exps/10,10_imagenet100.json',
    #                     help='Json file of settings.')
	
    return parser


if __name__ == '__main__':
	main()