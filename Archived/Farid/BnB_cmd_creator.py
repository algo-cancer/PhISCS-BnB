import numpy as np

with open('BnB_cmds.sh', 'w') as fout:
  for n in [20, 40, 80]:
    for m in [20, 40, 80]:
      for s in [10]:
        # for k in np.linspace(start=30, stop=int(n*m/3), num=10):
        for k in np.arange(start=30, stop=min(300, int(n*m/3)), step=10):
          k = int(k)
          cmd = 'cd BnB; '
          cmd += f'python cmp_algs.py -n {n} -m {m} -i 100 -s {s} -k {k} -t {int(np.sqrt(n*m)*k/10)} --source_type 3 --save_results\n'
          fout.write(cmd)
