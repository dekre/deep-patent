import hashlib
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


class FeatureScope(Enum):
    DB = "db"
    QRY = "qry"


def __is_cached(image_list: List[str], cache_dir: str, scope: FeatureScope) -> bool:
    uuid = __calc_features_hash(image_list)
    features_path = __get_cached_features_path(cache_dir, uuid, scope)
    return features_path.is_file()


def cache_features(
    features: torch.Tensor, image_list: List[str], cache_dir: str, scope: FeatureScope
) -> Path:
    uuid = __calc_features_hash(image_list)
    features_path = __get_cached_features_path(cache_dir, uuid, scope)    
    _ = np.save(features_path, features.cpu().numpy())
    return features_path


def load_cached_features(image_list: List[str], cache_dir: str, scope: FeatureScope) -> Optional[torch.Tensor]:
    if not __is_cached(image_list, cache_dir, scope):
        return None
    uuid = __calc_features_hash(image_list)
    features_path = __get_cached_features_path(cache_dir, uuid, scope)
    features = np.load(features_path)
    return torch.from_numpy(features)


def __get_cached_features_path(cache_dir: str, uuid: str, scope: FeatureScope) -> Path:
    scope = FeatureScope(scope)
    features_dir = Path(cache_dir).joinpath(scope.value)
    features_dir.mkdir(parents=True, exist_ok=True)
    return features_dir.joinpath(f"feats.{uuid}.npy")


def __calc_features_hash(image_list: List[str], sep: str = "_x_") -> str:
    return hashlib.md5(f"{sep}".join(image_list).encode("utf-8")).hexdigest()
    