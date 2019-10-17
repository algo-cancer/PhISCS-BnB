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
4. Check that everything works by running:
```bash
python compare_algorithms.py --help
```

## Preparing compare_algorithms.py
There are a few parameters you would want to set to run the experiment.


### In the command line:
- Number of cells:  -n N
- Number of columns:  -m M
- Number of repitions for each configuration: -i I
- You can use inputs that are
     (0) uniformly distributed random binary matrix
     (1) simulated and randomly added noise (will use -k argument too)
     (2) a fixed matrix (will use --instance_index too.
     Use the index (0-2) with -s SOURCE_TYPE, --source_type SOURCE_TYPE.
  -k K if SOURCE_TYPE=1, 
        if K is between 0 and 1 : the probablity of each 0 being flipped to 1.
        if K is bigger than 1: "the probablity of each 0 being flipped to 1" will set to K/number_zeros.
       otherwise ignored.
  --instance_index INSTANCE_INDEX
        if SOURCE_TYPE=2, a matrix in file Utils.instances is used with index INSTANCE_INDEX
        otherwise ignored.
- How long to allow BnB algorithms to run:  -t TIME_LIMIT, --time_limit TIME_LIMIT
- Booleans for reporting results:
  --print_rows
  --print_results
  --save_results

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

To loop through different values of m and n, you can not give any value in the arguments and then the lists of different values for n, m, number of repetitions can be set:
```
  iterList = itertools.product([ 70, 80, 90, 100, 120], # n
                               [ 80], # m
                               list(range(3)), # i
                               list(range(len(methods)))
                               )
```
The above example will try 70, 80, 90, 100, 120 for n when m is fixed at 80 and each one will be run 3 times.

### Done!
