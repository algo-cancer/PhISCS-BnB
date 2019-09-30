import platform # For the name of host machine
import getpass # For the username running the program
import sys


# For users and platforms
userName = getpass.getuser()
platformName = platform.node()
# End of all users


if userName == "esadeqia":
  import scipy.sparse as sp
  import numpy as np
  import pybnb
  import random
  import math
  import time
  import os
  import subprocess
  import numpy as np
  from gurobipy import *

  print(f"Running on {userName}@{platformName}")
  sys.path.append('/home/esadeqia/PhISCS_BnB/Utils')
  sys.path.append('/home/esadeqia/PhISCS_BnB')
  csp_solver_path = './openwbo'
  # ms_path = '/home/frashidi/software/bin/ms'
  ms_path = None
elif userName == "frashidi":
  csp_solver_path = './openwbo'
  ms_path = '/home/frashidi/software/bin/ms'
else:
  print("User const not found!")
  exit(0)