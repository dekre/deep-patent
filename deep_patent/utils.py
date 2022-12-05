from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch

import dirtorch.nets as nets
from dirtorch.utils import common
from dirtorch.utils.common import matmul, pool, tonumpy


def load_model(
    path: str, arch: str, iscuda: bool = False, image_size: int = 224
) -> torch.nn.Module:
    checkpoint = common.load_checkpoint(path, iscuda)

    # Set model options
    model_options = {}
    model_options["arch"] = arch
    model_options["without_fc"] = True

    net = nets.create_model(pretrained="", **model_options)
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint["state_dict"])

    net.preprocess = dict(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], input_size=image_size
    )

    if "pca" in checkpoint:
        net.pca = checkpoint.get("pca")
    return net


def whiten_db_and_qry_feats(
    net, whiten: dict, bdescs: torch.Tensor, qdescs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    bdescs = common.whiten_features(tonumpy(bdescs), net.pca, **whiten)
    qdescs = common.whiten_features(tonumpy(qdescs), net.pca, **whiten)
    return bdescs, qdescs
