from yacs.config import CfgNode as CN

_C = CN()
_C.PROJECT_NAME = "project_name"
_C.DATASET_NAME = "dataset_name"

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.NUM_WORKERS = 4

_C.IMAGE = CN()
_C.IMAGE.HEIGHT = 256
_C.IMAGE.WIDTH = 256
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "log"