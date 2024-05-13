import sys
import pathlib

module_path = pathlib.Path(__file__, "..", "../..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

import hydra
from omegaconf import OmegaConf




@hydra.main(config_name="config", version_base=None, config_path="config")
def main(cfg:config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()