import os, logging

def make_logger(p, str_name):
    """ make logging.getLogger

    Args:
        p (instance): Instance of 'parameters' class.
        str_name (str): Strings which will be the name of log file.

    Returns:
        logger (instance): Instance of 'logging.logger'
    """
    logger = logging.getLogger(str_name)
    logger.setLevel(logging.DEBUG)
    logfile_path = os.path.join(p.log_dir,"log_"+str_name+".dat")
    handler = logging.FileHandler(logfile_path)
    format = '%(asctime)s [%(levelname)s] %(name)s, lines %(lineno)d : %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format, date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger