

from .logger import setup_logger
import argparse
import os


def default_argument_parser():
    parser = argparse.ArgumentParser(description="")
    
    #parser.add_argument()
    
    parser.add_argument("--config_file", default="config/config.yaml", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    return parser


def project_preprocess(cfg):
    
    args = default_argument_parser().parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger = setup_logger(cfg.PROJECT_NAME, output_dir, 0)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    return cfg