

from .logger import set_logger,get_logger
import argparse
import os


def default_argument_parser():
    parser = argparse.ArgumentParser(description="")
    
    #parser.add_argument()
    
    parser.add_argument("--config-file", default="./configs/DCGAN.yaml" ,help="path to config file", type=str)
    parser.add_argument("--generator",'-g',help='path to generator model weights')
    parser.add_argument('--separate','-s',action='store_true',help='whether to separate the images')
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    return parser


def project_preprocess(cfg):
    
    args = default_argument_parser().parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.generator:
        cfg.MODEL.G.PATH = args.generator
    if args.separate:
        cfg.IMAGE.SEPARATE = True
    cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    set_logger(cfg)
    logger = get_logger()
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg