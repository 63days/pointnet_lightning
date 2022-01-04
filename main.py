from typing import List
from pytorch_lightning import Callback, seed_everything
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config.yaml")
def main(config: DictConfig):

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    model = hydra.utils.instantiate(config.model)
    datamodule = hydra.utils.instantiate(config.datamodule)

    callbacks: List[Callback] = []
    for _, cb_conf in config.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    for _, lg_conf in config.logger.items():
        logger.append(hydra.utils.instantiate(lg_conf))

    trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()
