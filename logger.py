import logging

logger = logging.getLogger("test")
logger.setLevel(level=logging.INFO)


def create_logger(name: str):
    log_format = logging.Formatter(
        fmt=f"%(levelname)s %(asctime)s (%(relativeCreated)d) \t %(pathname)s %(funcName)s L%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_file = logging.FileHandler(filename="logs/{}.log".format(name))
    log_file.setFormatter(log_format)
    log_file.setLevel(level=logging.INFO)

    logger.addHandler(log_file)

    return logger
