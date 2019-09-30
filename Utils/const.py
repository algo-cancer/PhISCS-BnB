import platform # For the name of host machine
import getpass # For the username running the program
import sys


# For users and platforms
userName = getpass.getuser()
platformName = platform.node()
# End of all users


if userName == "esadeqia":
  # dont put anything file that we have written ourselves
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
  import pybnb
  import operator
  import networkx as nx
  import copy
  import pandas as pd
  from tqdm import tqdm

  print(f"Running on {userName}@{platformName}")
  sys.path.append('/home/esadeqia/PhISCS_BnB/Utils')
  sys.path.append('/home/esadeqia/PhISCS_BnB')
  csp_solver_path = './Utils/openwbo'
  # ms_path = '/home/frashidi/software/bin/ms'
  ms_path = None
  output_folder_path = "/home/esadeqia/PhISCS_BnB/reports"
elif userName == "frashidi":
  csp_solver_path = './openwbo'
  ms_path = '/home/frashidi/software/bin/ms'
else:
  print("User const not found!")
  exit(0)