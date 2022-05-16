from yacs.config import CfgNode as CN

_C = CN()
_C.PROJECT_NAME = "project_name"

_C.DATASET = CN()
_C.DATASET.NAME = "dataset/name"
_C.DATASET.TRAIN_TEST_RATIO = 0.8

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.NUM_WORKERS = 0

_C.IMAGE = CN()  
_C.IMAGE.HEIGHT = 64
_C.IMAGE.WIDTH = 64
_C.IMAGE.CHANNEL = 3
_C.IMAGE.PIXEL_MEAN = (0.5, 0.5, 0.5)
_C.IMAGE.PIXEL_STD = (0.5, 0.5, 0.5)
_C.IMAGE.NUMBER = 10 # number of images to be recorded
_C.IMAGE.SAVE_NUMBER = 64 # number to save images
_C.IMAGE.SAVE_ROW_NUMBER = 8 # row to save images
_C.IMAGE.SAVE_PATH = "./images" # path to save images while running generate.py
_C.IMAGE.SEPARATE = False # if True, save images in separate

_C.MODEL = CN()
_C.MODEL.NAME = "model_name"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.CHECKPOINT_DIR = "checkpoints"

_C.MODEL.D = CN()
_C.MODEL.D.DIMENSION = 256
_C.MODEL.D.PATH = ""

_C.MODEL.G = CN()
_C.MODEL.G.DIMENSION = 1024 # DCGAN paper uses 1024, do not change if nesssary
_C.MODEL.G.INPUT_SIZE = 100 # DCGAN paper uses 100, do not change if nesssary
_C.MODEL.G.PATH = ""

# WGAN arguments
_C.MODEL.WGAN = CN()
_C.MODEL.WGAN.WEIGHT_CLIPING_LIMIT = 0.01
_C.MODEL.WGAN.GENERATOR_ITERS = 40000 # iteration for WGAN
_C.MODEL.WGAN.CRITIC_ITERS = 5
_C.MODEL.WGAN.LAMBDA = 10
_C.MODEL.WGAN.IC = False

_C.WALKING_LATENT_SPACE = CN()
_C.WALKING_LATENT_SPACE.STEP = 50
_C.WALKING_LATENT_SPACE.IMAGE_NUMBER = 16
_C.WALKING_LATENT_SPACE.IMAGE_ROW_NUMBER = 4
_C.WALKING_LATENT_SPACE.IMAGE_FPS = 10

_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 0.00001
_C.SOLVER.BETAS = (0.9, 0.999)
_C.SOLVER.EPOCHS = 300 # not used in WGAN, for DCGAN
_C.SOLVER.CHECKPOINT_FREQ = 500
_C.SOLVER.EVALUATE_ITERATION = 125
_C.SOLVER.EVALUATE_BATCH = 128
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "log"
_C.LOG_CONFIGURATION = "config/logging.conf"