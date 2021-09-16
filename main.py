import argparse
import src.utils.config as config

def main():
	arg_parse = argparse.ArgumentParser(description="")
	arg_parser.add_argument(
		'config',
		metavar='config_json_file',
		default='None',
		help='The Configuration file in json format')
	args = arg_parser.parse_args()

	config = config.process_config(args.config)



if __name__ == '__main__':
	main()