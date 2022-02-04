import click
import cont_assoc.datasets.kitti_dataset as kitti_dataset
import cont_assoc.models.ps4d_models as models
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
@click.option('--test_set', is_flag=True)
@click.option('--ckpt_ps', type=str, required=True)
@click.option('--ckpt_ag', type=str, required=True)
@click.option('--save', is_flag=True)
@click.option('--config_ps', '-cs', type=str,
              default=join(getDir(__file__), '../config/panoptic_cylinder.yaml'))
@click.option('--config_ag', '-tr', type=str,
              default=join(getDir(__file__), '../config/contrastive_instances.yaml'))
def main(config_ps, config_ag, ckpt_ps, ckpt_ag, save, test_set):
    ps_cfg = edict(yaml.safe_load(open(config_ps)))
    ag_cfg = edict(yaml.safe_load(open(config_ag)))

    if save:
        results_dir = create_dirs(seq, test_set)
        ps_cfg.RESULTS_DIR = results_dir

    ps_cfg.UPDATE_METRICS = 'True'

    ckpt_ps_path = join(getDir(__file__), ckpt_ps)
    checkpoint_ps = torch.load(ckpt_ps_path, map_location='cpu')
    ckpt_ag_path = join(getDir(__file__), ckpt_ag)
    checkpoint_ag = torch.load(ckpt_ag_path, map_location='cpu')

    model = models.PS4D(ps_cfg, ag_cfg)
    model.load_state_dicts(checkpoint_ps['state_dict'],checkpoint_ag['state_dict'])

    data = kitti_dataset.SemanticKittiModule(ps_cfg)
    data.setup()

    trainer = Trainer(gpus=ps_cfg.EVAL.N_GPUS, logger=False)

    print("Setup finished, starting evaluation")

    if test_set:
        trainer.test(model,data.test_dataloader())
    else:
        trainer.test(model,data.val_dataloader())

    model.panoptic_model.evaluator.print_results()
    print("#############################################################")
    model.evaluator4D.calculate_metrics()
    model.evaluator4D.print_results()

def create_dirs(seq, test_set):
    results_dir = join(getDir(__file__), 'output', 'val', '4d_ps', 'sequences')
    if test_set:
        results_dir = join(getDir(__file__), 'output', 'test', '4d_ps', 'sequences')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    if seq:
        sub_dir = os.path.join(results_dir, str(seq).zfill(2), 'predictions')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    elif test_set:
        for i in range(11,22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)
    else:
        sub_dir = os.path.join(results_dir, str(8).zfill(2), 'predictions')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir, exist_ok=True)
    return results_dir


if __name__ == "__main__":
    main()
