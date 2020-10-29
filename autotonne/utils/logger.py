import logging
from logging.handlers import RotatingFileHandler


class LogFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def setup_logger():
    MAX_BYTES = 10000000  # Maximum size for a log file
    BACKUP_COUNT = 9  # Maximum number of old log files

    # The name should be unique, so you can get in in other places
    # by calling `logger = logging.getLogger('com.logger.example')
    logger = logging.getLogger('com.logger.example')
    logger.setLevel(logging.INFO)  # the level should be the lowest level set in handlers

    log_format = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    info_handler = RotatingFileHandler('info.log', maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    info_handler.setFormatter(log_format)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(LogFilter(logging.INFO))
    logger.addHandler(info_handler)

    error_handler = RotatingFileHandler('error.log', maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
    error_handler.setFormatter(log_format)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    return logger


LOGGER = setup_logger()
