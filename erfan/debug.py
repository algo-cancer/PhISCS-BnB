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
from lp_bounding import makeGurobiModel, flip, unFlipLast, LP_Bounding_Model, LP_brief
import copy
import scipy.sparse as sp

if __name__ == '__main__':
  x = np.array([[1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 0, 1, 1, 1, 0],
                [1, 0, 1, 1, 0, 1]], dtype=np.int8)

  delta = np.array([[0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0]], dtype=np.int8)

  xp = x + delta

  nf = myPhISCS_I(x)

  res = is_conflict_free_gusfield_and_get_two_columns_in_coflicts(xp)


  print(res, nf, )