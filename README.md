## Installation

1. You need to install pybnb.

2. If you can install gurobi it would be great because in the script we are comparing the running time with gurobi as well. If you don't want to, simply get rid of the following lines:

```
from utils import *
solution, (flips_0_1, flips_1_0, flips_2_0, flips_2_1) = PhISCS_I(noisy, beta=0.9, alpha=0.00000001)
```

## Guideline

The script creates a random matrix sized by n and m. Then it runs PhISCS_I (Gurobi) and reports the minimum number of flips solved by Gurobi in addition to the time needed. Finally, the script runs Phylogeny BnB (based on pybnb) and reports how much time is needed in addition to the number of flips.

## Usage

For running you can use the following commands:

```
python phylogeny_bnb.py -n 5 -m 5 -w c -r
```

-n is the number of cells  
-m is the number of mutations  
-w indicates which heuristic algorithm is being used (a single character in {a,b,c,d})  
-r shows whether a random partitioning should be used or not.

I would recommend using -w either b or c.

You may use the following commands:

```
python phylogeny_bnb.py -n 10 -m 10 -w c -r
python phylogeny_bnb.py -n 7 -m 5 -w c
python phylogeny_bnb.py -n 10 -m 7 -w b -r
python phylogeny_bnb.py -n 15 -m 8 -w b
```
