"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    print(cfg.exp_name)


if __name__ == '__main__':
    main()
