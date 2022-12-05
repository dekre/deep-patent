### ©2020. Triad National Security, LLC. All rights reserved.
### This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
###
###

import json
import os
import pathlib
from typing import Union

import numpy as np
import torch
import tqdm

from dirtorch.datasets.generic import ImageListLabelsQ
from dirtorch.utils import common

from .utils import load_model

fld = str(pathlib.Path(__file__).parent.resolve())
if fld.startswith("/vast"):
    fld = fld.replace("/vast", "")


class DeepPatentTest(ImageListLabelsQ):
    def __init__(self):
        ImageListLabelsQ.__init__(
            self,
            img_list_path=os.path.join(fld, "data/test_db_patent.txt"),
            query_list_path=os.path.join(fld, "data/test_query_patent.txt"),
            root=args.dataset,
        )


def eval_query(
    db_images: ImageListLabelsQ,
    scores: Union[np.ndarray, torch.Tensor],
    detailed: bool = True,
):
    map_res = __compute_map(db_images=db_images, scores=scores, detailed=detailed)
    top_res = __compute_top(db_images=db_images, scores=scores, detailed=detailed)
    return {**map_res, **top_res}


def __compute_top(
    db_images: ImageListLabelsQ,
    scores: Union[np.ndarray, torch.Tensor],
    detailed: bool = True,
) -> dict:
    res = dict()
    try:
        tops = [
            db_images.eval_query_top(q, s)
            for q, s in enumerate(tqdm.tqdm(scores, desc="top1"))
        ]
    except NotImplementedError:
        pass
    if detailed:
        res["tops"] = tops
    for k in tops[0]:
        res["top%d" % k] = float(np.mean([top[k] for top in tops]))


def __compute_map(
    db_images: ImageListLabelsQ,
    scores: Union[np.ndarray, torch.Tensor],
    detailed: bool = True,
):
    res = dict()
    try:
        aps = [
            db_images.eval_query_AP(q, s)
            for q, s in enumerate(tqdm.tqdm(scores, desc="AP"))
        ]
    except NotImplementedError:
        print(" AP not implemented!")

    if not isinstance(aps[0], dict):
        aps = [float(e) for e in aps]
        if detailed:
            res["APs"] = aps
        # Queries with no relevants have an AP of -1
        res["mAP"] = float(np.mean([e for e in aps if e >= 0]))
    else:
        modes = aps[0].keys()
        for mode in modes:
            apst = [float(e[mode]) for e in aps]
            if detailed:
                res["APs" + "-" + mode] = apst
            # Queries with no relevants have an AP of -1
            res["mAP" + "-" + mode] = float(np.mean([e for e in apst if e >= 0]))
    return res


if __name__ == "__main__":

    args = parser.parse_args()

    """ Define the dataset class
    """

    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None:
        args.aqe = {"k": args.aqe[0], "alpha": args.aqe[1]}
    if args.adba is not None:
        args.adba = {"k": args.adba[0], "alpha": args.adba[1]}

    dataset = DeepPatentTest()
    print("Test dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda, args)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {
            "whitenp": args.whitenp,
            "whitenv": args.whitenv,
            "whitenm": args.whitenm,
        }
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = test_dir.eval_model(
        dataset,
        net,
        args.trfs,
        pooling=args.pooling,
        gemp=args.gemp,
        detailed=args.detailed,
        threads=args.threads,
        dbg=args.dbg,
        whiten=args.whiten,
        aqe=args.aqe,
        adba=args.adba,
        save_feats=args.save_feats,
        load_feats=args.load_feats,
    )

    if not args.detailed:
        print(" * " + "\n * ".join(["%s = %g" % p for p in res.items()]))

    if args.out_json:
        # write to file
        try:
            data = json.load(open(args.out_json))
        except IOError:
            data = {}
        data[args.dataset] = res
        mkdir(args.out_json)
        open(args.out_json, "w").write(json.dumps(data, indent=1))
        print("saved to " + args.out_json)
