import logging
from termcolor import colored

DEBUG = logging.debug
INFO = lambda s: logging.info(s)
DONE = lambda s: logging.info(colored(s, 'green'))
FAIL = lambda s: logging.warning(colored(s, 'yellow'))