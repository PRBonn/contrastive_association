import click
import cont_assoc.datasets.kitti_dataset as kitti_dataset
import cont_assoc.models.panoptic_models as models
from easydict import EasyDict as edict
import os
from os.path import join
from pytorch_lightning import Trainer
import torch
import yaml

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

@click.command()
@click.option('--ckpt', type=str, required=True)
@click.option('--save_val_pred', is_flag=True)
@click.option('--config', '-c', type=str,
              default=join(getDir(__file__), '../config/panoptic_cylinder.yaml'))
def main(config, ckpt, save_val_pred):
    cfg = edict(yaml.safe_load(open(config)))

    cfg.DATA_CONFIG.DATALOADER.SHUFFLE = False

    if save_val_pred:
        cfg.SAVE_VAL_PRED = 'True'
    else:
        cfg.SAVE_FEATURES = 'True'

    ckpt_path = join(getDir(__file__), ckpt)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    model = models.PanopticCylinder(cfg)
    model.load_state_dict(checkpoint['state_dict'])

    data = kitti_dataset.SemanticKittiModule(cfg)
    data.setup()

    trainer = Trainer(gpus=cfg.EVAL.N_GPUS, logger=False)

    if save_val_pred:
        trainer.test(model,data.val_dataloader())
    else:
        trainer.test(model,data.train_dataloader())

if __name__ == "__main__":
    main()
