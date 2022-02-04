import click
import cont_assoc.datasets.ins_feat_dataset as ins_dataset
import cont_assoc.models.contrastive_models as models
from easydict import EasyDict as edict
import os
from os.path import join
from pytorch_lightning import Trainer
import subprocess
import torch
import yaml

def getDir(obj):
    return os.path.dirname(os.path.abspath(obj))

@click.command()
@click.option('--ckpt', type=str, required=True)
@click.option('--config', '-c', type=str,
              default=join(getDir(__file__), '../config/contrastive_instances.yaml'))
@click.option('--seq',
              type=int,
              default=None,
              required=False)
def main(config, ckpt, seq):
    cfg = edict(yaml.safe_load(open(config)))

    if seq:
        cfg.TRAIN.ONLY_SEQ = seq

    cfg.UPDATE_METRICS = 'True'

    ckpt_path = join(getDir(__file__), ckpt)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    model = models.ContrastiveTracking(cfg)
    model.load_state_dict(checkpoint['state_dict'])

    data = ins_dataset.InstanceFeaturesModule(cfg)
    data.setup()

    trainer = Trainer(gpus=cfg.TRAIN.N_GPUS, logger=False)

    trainer.test(model,data.val_dataloader())

    if not seq:
        model.evaluator4D.calculate_metrics()
        model.evaluator4D.print_results()
        AQ = model.evaluator4D.get_mean_aq()
        print("AQ",AQ)

if __name__ == "__main__":
    main()
