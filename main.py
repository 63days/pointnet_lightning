from typing import List
from pytorch_lightning import Callback, seed_everything
import hydra
from omegaconf import DictConfig
from utils import print_config


@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    
    # print all configs
    print_config(config)

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    model = hydra.utils.instantiate(config.model)
    datamodule = hydra.utils.instantiate(config.datamodule)

    callbacks: List[Callback] = []
    if config.get("callbacks"):
        for _, cb_conf in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if config.get("logger"):
        for _, lg_conf in config.logger.items():
            logger.append(hydra.utils.instantiate(lg_conf))

    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()
