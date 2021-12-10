"""
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse

from utils.config import process_config
from agent.vbl_agent import VBLAgent

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    arg_parser.add_argument(
        '--mode',
        metavar='mode_of_running',
        default='train',
        help='Mode of running: train or test')
    arg_parser.add_argument(
        '--wandb',
        metavar='wanb_logging',
        default='true',
        help='Enter your wandb setting')
    arg_parser.add_argument(
        '--model',
        metavar='model_name',
        default='unet',
        help='Enter name of the Model')
    arg_parser.add_argument(
        '--weighted',
        metavar='weighted_training',
        default='false',
        help='Weighted training mode or not')
    

    args = arg_parser.parse_args()

    # Parse the config json file
    config = process_config(args.config)

    # Set mode provided
    config.mode = args.mode
    config.wandb = args.wandb
    config.model = args.model
    config.weighted = args.weighted

    # Create the Agent and pass all the configuration to it then run it..
    agent = VBLAgent(config)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()