## Installation
1. Make sure you have a working version of Gurobi in your system by following the steps explained here: https://www.gurobi.com/documentation/8.1/quickstart_linux/installing_the_anaconda_py.html
2. Download the project from gitHub with
```bash
git clone https://github.com/faridrashidi/Phylogeny_BnB.git
```
3. Install all python requirments via:
```bash
pip install -r requirments.txt
```
4. Check everything works by running:
```bash
python compare_algorithms.py
```

## Preparing compare_algorithms.py
There are a few parameters you would want to set to run the experiment.

### At the top we have:
```
timeLimit = 300
queue_strategy = "custom"
sourceType = ["RND",
              "MS",
              "FIXED"][1]
```
The last one indicates if simulated matrices should be used or totally random matrices. Also you can feed in a specific matrix with the last option.

### Inside main script:
A list of methods need to be indicated for ''methods''. E.g., 
```
 methods = [
    (PhISCS_I, None),
    (PhISCS_B, None),
    ("OldBnB", lb_lp_gurobi),
    ("BnB", SemiDynamicLPBounding(ratio=None, continuous = True, tool = "Gurobi", prioritySign = -1)),
]
```

In the iterList, the lists of different values for n, m, number of repetitions need to be entered:
```
  iterList = itertools.product([ 70, 80, 90, 100, 120], # n
                               [ 80], # m
                               list(range(3)), # i
                               list(range(len(methods)))
                               )
```
The above example will try 70, 80, 90, 100, 120 for n when m is fixed at 80 and each one will be run 3 times.

### Done!
