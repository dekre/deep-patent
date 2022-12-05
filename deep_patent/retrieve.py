from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import tqdm

from dirtorch.datasets.generic import ImageList, ImageListLabels
from dirtorch.utils import common
from dirtorch.utils.common import matmul, pool
from dirtorch.utils.pytorch_loader import get_loader

from .cache import FeatureScope, cache_features, load_cached_features
from .utils import whiten_db_and_qry_feats


def extract_image_features(
    dataset: Union[ImageList, ImageListLabels],
    transforms: List[str],
    net: torch.nn.Module,
    ret_imgs: bool = False,
    same_size: bool = False,
    desc: str = "Extract feats...",
    iscuda: bool = True,
    threads: int = 8,
    batch_size: int = 8,
):
    """Extract image features for a given dataset.
    Output is 2-dimensional (B, D)
    """
    if not same_size:
        batch_size = 1
        old_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

    loader = get_loader(
        dataset,
        trf_chain=transforms,
        preprocess=net.preprocess,
        iscuda=iscuda,
        output=["img"],
        batch_size=batch_size,
        threads=threads,
        shuffle=False,
    )

    if hasattr(net, "eval"):
        net.eval()

    tocpu = (lambda x: x.cpu()) if ret_imgs == "cpu" else (lambda x: x)

    img_feats = []
    trf_images = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(
            loader, desc, total=1 + (len(dataset) - 1) // batch_size
        ):
            # TODO this the inportant part!!!
            imgs = inputs[0]
            imgs = common.variables(inputs[:1], net.iscuda)[0]
            desc = net(imgs)
            if ret_imgs:
                trf_images.append(tocpu(imgs.detach()))
            del imgs
            del inputs
            if len(desc.shape) == 1:
                desc.unsqueeze_(0)
            img_feats.append(desc.detach())

    img_feats = torch.cat(img_feats, dim=0)
    if len(img_feats.shape) == 1:
        img_feats.unsqueeze_(0)

    if not same_size:
        torch.backends.cudnn.benchmark = old_benchmark

    if ret_imgs:
        if same_size:
            trf_images = torch.cat(trf_images, dim=0)
        return trf_images, img_feats
    return img_feats


def __compute_features_for_images(
    image_list: ImageList,
    net: torch.nn.Module,
    scope: Union[FeatureScope, str],
    threads: int = 8,
    trfs: Union[str, List[str]] = "",
    batch_size: int = 16,
    pooling="gemp",
    gemp=3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    trfs_list = [trfs] if isinstance(trfs, str) else trfs
    scope = FeatureScope(scope)
    image_feature_list: List[torch.Tensor] = []
    for trfs in trfs_list:
        kwargs = dict(
            iscuda=net.iscuda,
            threads=threads,
            batch_size=batch_size,
            same_size="Pad" in trfs or "Crop" in trfs,
        )
        image_feature_list.append(
            extract_image_features(image_list, trfs, net, desc=scope.value, **kwargs)
        )
    # pool from multiple transforms (scales)
    image_features = F.normalize(pool(image_feature_list, pooling, gemp), p=2, dim=1)
    return image_features


def compute_similarity_scores(
    qry_images: ImageList,
    db_images: ImageList,
    net: torch.nn.Module,
    threads: int = 8,
    trfs: str = "",
    batch_size: int = 16,
    whiten: dict = {},
    pooling="gemp",
    gemp=3,
    cache_dir: Optional[str] = ".cache",
):
    qdescs, bdescs = None, None
    if cache_dir:
        qdescs = load_cached_features(
            qry_images.imgs, cache_dir, scope=FeatureScope.QRY
        )
        bdescs = load_cached_features(db_images.imgs, cache_dir, scope=FeatureScope.DB)
    if qdescs is None:
        qdescs = __compute_features_for_images(
            image_list=qry_images,
            net=net,
            scope=FeatureScope.QRY,
            threads=threads,
            trfs=trfs,
            batch_size=batch_size,
            pooling=pooling,
            gemp=gemp,
        )
    if qdescs is None:
        bdescs = __compute_features_for_images(
            image_list=db_images,
            net=net,
            scope=FeatureScope.DB,
            threads=threads,
            trfs=trfs,
            batch_size=batch_size,
            pooling=pooling,
            gemp=gemp,
        )

    qdescs = torch.nan_to_num(qdescs)
    bdescs = torch.nan_to_num(bdescs)

    if cache_dir:
        _ = cache_features(
            features=qdescs,
            image_list=qry_images.imgs,
            cache_dir=cache_dir,
            scope=FeatureScope.QRY,
        )
        _ = cache_features(
            features=bdescs,
            image_list=db_images.imgs,
            cache_dir=cache_dir,
            scope=FeatureScope.DB,
        )

    if whiten:
        bdescs, qdescs = whiten_db_and_qry_feats(
            net, whiten, bdescs=bdescs, qdescs=qdescs
        )

    scores = matmul(qdescs, bdescs)

    del bdescs
    del qdescs

    return scores
