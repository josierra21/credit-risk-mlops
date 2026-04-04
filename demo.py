from credit_risk.logger import logging
from credit_risk.exception import CREDITriskException
import sys


logging.info("Welcome to our custom log")

try: 
    a = 2/0
except Exception as e:
    raise CREDITriskException(e, sys)