import os, logging

def make_logger(p, name_str):
    """ make logging.getLogger

    Args:
        p (instance): Instance of 'parameters' class.
        str_name (str): Strings which will be the name of log file.

    Returns:
        logger (instance): Instance of 'logging.logger'
    """
    logger = logging.getLogger(name_str)
    logger.setLevel(logging.DEBUG)
    logfile_path = os.path.join(p.log_dir,"log_"+name_str+".dat")
    if os.path.isfile(p.log_dir+"/log_"+name_str+".dat"): os.system("rm "+p.log_dir+"/log_"+name_str+".dat")
    handler = logging.FileHandler(logfile_path)
    format = '%(asctime)s [%(levelname)s] %(name)s, lines %(lineno)d : %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format, date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger