from yacs.config import CfgNode as CN

_C = CN()
_C.PROJECT_NAME = "project_name"

_C.DATASET = CN()
_C.DATASET.NAME = "dataset/name"
_C.DATASET.TRAIN_TEST_RATIO = 0.8

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.NUM_WORKERS = 4

_C.IMAGE = CN()  
_C.IMAGE.HEIGHT = 32
_C.IMAGE.WIDTH = 32
_C.IMAGE.CHANNEL = 3
_C.IMAGE.PIXEL_MEAN = (0.5, 0.5, 0.5)
_C.IMAGE.PIXEL_STD = (0.5, 0.5, 0.5)
_C.IMAGE.NUMBER = 10 # number of images to be recorded

_C.MODEL = CN()
_C.MODEL.NAME = "model_name"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.CHECKPOINT_DIR = "checkpoints"
_C.MODEL.SAVE_FREQ = 1

_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.WEIGHT_DECAY = 0.00001
_C.SOLVER.BETAS = (0.5, 0.999)
_C.SOLVER.EPOCHS = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "log"