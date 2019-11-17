Experiment checklist:
- ILP behaviour
 - making matrices 100x100 in PHI
 - then I should run those in bigred


- Hybrid of MWM and SemiLP
  - priority queue
  
Ongoing experiments:


- Correct strength function for subsample
  - subsamp_bin_search.py running in phi
    - Nov16 138/180 77%
  
  
  
Old notes:
- Memory limit exceeds 10gb for Phisics_I : 'n': '100', 'm': '100', 'k': '40' in carbonate
  - exceeded even with 16gb
  - I am gonna try Karst
  - This is solved in BigRed3

  
  
  
  
  Install the project:
  
- conda create -n env2 python=3.6
- source activate env2
- conda install -c http://conda.anaconda.org/gurobi gurobi
- module load openmpi/gnu/2.1.0
- pip install mpi4py
- export LC_CTYPE=en_US.UTF-8
- pip install numpy
- pip install python-sat
- pip install py-aiger-cnf
- pip install -r requirments_no_version.txt
- python -m Utils.const



copying the noisy files
tar -czf noisy_nov16.tar.gz noisy

in bigred:
scp esadeqia@phi.cs.indiana.edu:/home/esadeqia/PhISCS_BnB/Data/noisy_nov16.tar.gz .
tar -xzf noisy_nov16.tar.gz
