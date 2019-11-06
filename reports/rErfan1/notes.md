Experiment checklist:
- Memory limit exceeds 10gb for Phisics_I : 'n': '100', 'm': '100', 'k': '40' in carbonate
  - exceeded even with 16gb
  - I am gonna try Karst
  

Ongoing experiments:
- Previous methods: What is nature of relation with n_flip, n, m?
  - exp1_rnd_prevM_P (1437460)
  - exp1_ms_prevM_P (1437495): potential limit before
  - exp1_salem_prevM_P: choose correct combinations
- Hybrid of MWM and SemiLP

- Correct strength function for subsample
  - "find_relation_subsam.py" screen is running on Phi 70% on 10am nov 5th.
  - cmp_algs_100,100,10_11-03-16-03-46.csv seems bigger pool didn't help?
  
  
  
  
  
  Install the project:
  
conda create -n env2
source activate env2
conda install -c http://conda.anaconda.org/gurobi gurobi
module load openmpi/gnu/2.1.0
pip install mpi4py
# export sat stuff
pip install -r requirments.txt
