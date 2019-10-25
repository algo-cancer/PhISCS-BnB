import sys
sys.path.append('../..')
from Utils.util import *
import pandas as pd

simulation_folder_path_in = "/data/frashidi/BnB/simulations/"
simulation_folder_path_out = "/data/frashidi/BnB/output/"

for n in [10, 20, 40]:
  for m in [10, 20, 40]:
    cell_names = ['cell'+str(j) for j in range(n)]
    mut_names = ['mut'+str(j) for j in range(m)]      
    for s in [4, 10]:
      for k in np.linspace(start=10, stop=int(n*m/2), num=20):
        k = int(k)
        for i in range(0, 100):
          fin = f"simNo_{i+1}-s_{s}-m_{m}-n_{n}.SC.ground"
          file = simulation_folder_path_in + fin
          dfin = pd.read_csv(file, delimiter="\t", index_col=0)

          O = make_noisy_by_k(dfin.values, k)
          dfout = pd.DataFrame(O)
          dfout.columns = mut_names
          dfout.index = cell_names
          dfout.index.name = 'cellIDxmutID'

          fout = f"simNo_{i+1}-s_{s}-m_{m}-n_{n}-k_{k}.SC.noisy"
          file = simulation_folder_path_out + fout
          dfout.to_csv(file, sep='\t')
