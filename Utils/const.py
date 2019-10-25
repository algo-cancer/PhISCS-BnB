import copy
import datetime
import inspect
import getpass  # For the username running the program
import math
# import matplotlib as mpl
# mpl.use("Agg")
# import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import operator
import pandas as pd
import pickle
import platform  # For the name of host machine
import pybnb
import random
import scipy.sparse as sp
import subprocess
import sys
import time
import traceback

from collections import defaultdict
from gurobipy import *
from ortools.linear_solver import pywraplp
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from tqdm import tqdm



def print_line(depth=1):
    """A debugging tool!  """
    for i in range(1, depth + 1):
        info = inspect.stack()[i]
        for j in range(i - 1):
            print("\t", end="")
        print(f"Line {info.lineno} in {info.filename}, Function: {info.function}")


# This line is here to make sure the line "Academic license - for non-commercial use only" prints at the top
# gurobi_env = Env()
# For users and platforms
user_name = getpass.getuser()
platform_name = platform.node()
# End of all users

print(f"Running on {user_name}@{platform_name}")

if user_name == "esadeqia":
    openwbo_path = "/home/esadeqia/external/openwbo"
    ms_path = "/home/esadeqia/external/ms"
    output_folder_path = "/home/esadeqia/PhISCS_BnB/reports/Erfan"
    simulation_folder_path = "/home/esadeqia/PhISCS_BnB/simulations/"
    if "carbonate" in platform_name:
        openwbo_path = "/gpfs/home/e/s/esadeqia/Carbonate/external/openwbo"
        ms_path = "/gpfs/home/e/s/esadeqia/Carbonate/external/ms"
        output_folder_path = "/gpfs/home/e/s/esadeqia/Carbonate/Phylogeny_BnB/reports/Erfan"
        simulation_folder_path = "/home/esadeqia/PhISCS_BnB/simulations/"
elif user_name == "school":
    openwbo_path = None
    ms_path = None
    output_folder_path = "./reports/Erfan"
elif user_name == "frashidi":
    openwbo_path = "./Archived/Farid/archived/openwbo"
    ms_path = "./Archived/Farid/archived/ms"
    output_folder_path = "./reports"
    simulation_folder_path = "./simulations/"
else:
    print("User const not found!")
    exit(0)

