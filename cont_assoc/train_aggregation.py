import click
import cont_assoc.datasets.ins_feat_dataset as ins_dataset
import cont_assoc.models.contrastive_models as models
from easydict import EasyDict as edict
import os
from os.path import join
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import subprocess
import torch
import yaml

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

@click.command()
@click.option('--config',
              '-c',
              type=str,
              default=join(getDir(__file__), '../config/contrastive_instances.yaml'))
@click.option('--seq',
              type=int,
              default=None,
              required=False)
@click.option('--weights',
              type=str,
              default=None,
              required=False)
def main(config, seq, weights):
    cfg = edict(yaml.safe_load(open(config)))

    if seq is not None:
        cfg.TRAIN.ONLY_SEQ = seq

    data = ins_dataset.InstanceFeaturesModule(cfg)
    model = models.ContrastiveTracking(cfg)

    #Load pretrained weights
    if weights:
        pretrain = torch.load(weights, map_location='cpu')
        model.load_state_dict(pretrain['state_dict'],strict=True)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg.EXPERIMENT.ID,
                                             default_hp_metric=False)

    #Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint = ModelCheckpoint(monitor='AQ',
                                 filename=cfg.EXPERIMENT.ID+'_{epoch:03d}_{AQ:.3f}',
                                 mode='max',
                                 save_last=True)

    trainer = Trainer(gpus=cfg.TRAIN.N_GPUS,
                      logger=tb_logger,
                      max_epochs= cfg.TRAIN.MAX_EPOCH,
                      log_every_n_steps=10,
                      callbacks=[lr_monitor, checkpoint])

    trainer.fit(model, data)

if __name__ == "__main__":
    main()
