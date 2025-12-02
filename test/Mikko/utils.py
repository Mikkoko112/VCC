import anndata as ad
from pathlib import Path

from constants import DATAPATH

def read_data(file : str | Path = DATAPATH, backed : bool = False):
    adata = ad.read_h5ad(file, backed="r")
    return adata