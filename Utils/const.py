import sys
import platform  # For the name of host machine
import getpass  # For the username running the program
import scipy.sparse as sp
import pybnb
import random
import math
import time
import os
import subprocess
import numpy as np
from gurobipy import *
import datetime
from collections import defaultdict
import operator
import networkx as nx
import copy
import pandas as pd
from tqdm import tqdm
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import inspect
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

if True: # TODO: why this runs twice
  constHasRun = True
  # For users and platforms
  userName = getpass.getuser()
  platformName = platform.node()
  # End of all users

  print(f"Running on {userName}@{platformName}")
  # gurobi_env = Env()

  if userName == "esadeqia":
    sys.path.append('/home/esadeqia/PhISCS_BnB/Utils')
    sys.path.append('/home/esadeqia/PhISCS_BnB')
    openwbo_path = '/home/esadeqia/external/openwbo'
    ms_path = '/home/esadeqia/external/ms'
    output_folder_path = "/home/esadeqia/PhISCS_BnB/reports"
  elif userName == "frashidi":
    output_folder_path = "./reports"
    sys.path.append('./Utils')
    openwbo_path = './openwbo'
    ms_path = '/home/frashidi/software/bin/ms'
  else:
    print("User const not found!")
    exit(0)