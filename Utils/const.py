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

mpl.use("Agg")
import matplotlib.pyplot as plt
import pickle
from ortools.linear_solver import pywraplp


def print_line(depth=1):
    """A debugging tool!  """
    for i in range(1, depth + 1):
        info = inspect.stack()[i]
        for j in range(i - 1):
            print("\t", end="")
        print(f"Line {info.lineno} in {info.filename}, Function: {info.function}")


# This line is here to make sure the line "Academic license - for non-commercial use only" prints at the top
gurobi_env = Env()
# For users and platforms
user_name = getpass.getuser()
platform_name = platform.node()
# End of all users

print(f"Running on {user_name}@{platform_name}")

if user_name == "esadeqia":
    openwbo_path = "/home/esadeqia/external/openwbo"
    ms_path = "/home/esadeqia/external/ms"
    output_folder_path = "/home/esadeqia/PhISCS_BnB/reports/Erfan"
elif user_name == "school":
    openwbo_path = None
    ms_path = None
    output_folder_path = "./reports/Erfan"
elif user_name == "frashidi":
    openwbo_path = "/data/frashidi/Phylogeny_BnB/Archived/Farid/archived/openwbo"
    ms_path = "/home/frashidi/software/bin/ms"
    output_folder_path = "/data/frashidi/Phylogeny_BnB/reports"
else:
    print("User const not found!")
    exit(0)

