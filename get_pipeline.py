from freeprocess.todatawrapper import *
from freeprocess.proprocessor import *
from freeprocess.datatransformer import DataTransformer
from freeprocess.featurecross import NullPipeline, featurecross
from freeprocess.modeler import Modeler
from freeprocess.gpgenerate import GpGenerate


import logging
from logging import handlers

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger


def create_pipeline(enable_cross=True,enable_gp=True,class_balance=True):

    if enable_cross:
        cross_pipeline = featurecross
    else:
        cross_pipeline = NullPipeline()

    freemindpipeline = Pipeline(
    [
            ('preprocess',preprocess_pipeline),
            ('datatransform',DataTransformer()),
            ('cross',cross_pipeline),
            ('gp',GpGenerate(enable=enable_gp)),
            ('modeling',Modeler(use_raw_data=True,class_balance=class_balance))
        ]
    )

    return freemindpipeline
