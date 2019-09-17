import numpy as np
from instances import I1
from funcs import *
from utils import *
import operator
from collections import defaultdict
import time
import pandas as pd
from tqdm import tqdm
import itertools
from BnB import *
from t1 import gmLBw, gmLB

def randomPartitionBounding(I):
  return get_lower_bound(I, partition_randomly=True)


def greedyPartitionBounding(I):
  return get_lower_bound(I, partition_randomly=False)


def mxWeightedMatchingPartitionBounding(I):
  return gmLBw(I)

def mxMatchingPartitionBounding(I):
  return gmLB(I)
