import warnings, multiprocessing
from time import time
import numpy as np
from make_logger import make_logger
from parameters   import parameters
from hamiltonian  import hamiltonian

warnings.simplefilter('ignore')

if __name__=='__main__':
    s = time()
    p = parameters()
    logger = make_logger(p, "main")
    # self.logger.info("Number of Parallelization: {0}".format(p.core))

    time_s_ham = time()
    h = hamiltonian(p)
    logger.info("time = {}".format(time()-s))
