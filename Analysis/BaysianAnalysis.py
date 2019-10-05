import sys
if __name__ == '__main__':
  sys.path.append('../Utils')
  from const import *
elif "constHasRun" not in globals():
  from Utils.const import *

import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-f', '--filename', dest='filename', help=f'csvFileName in {output_folder_path}', type=str)
args = parser.parse_args()


saveTrace = True

if __name__ == '__main__':
  if "/" in args.filename:
    fileAddress = args.filename
    fileName = (args.filename.split("/")[-1]).split(".")[-1]
  else:
    fileAddress = os.path.join(output_folder_path, args.filename)
    fileName = args.filename
  print(f"Processing {fileAddress}")
  original_df = pd.read_csv(fileAddress)
  original_df.rename(columns = {"nf": "k"}, inplace = True)
  methods = ["PhISCS_I_"]
  predictor_names = ["n", "m", "k"]
  df = original_df[original_df["method"]==methods[0]]
  n_data = df.shape[0]
  y = df["runtime"]
  x = df[predictor_names]
  n_predictors = len(x.columns)

  print(n_data, n_predictors)
  print(x.head())
  print(y.head())
  print(x.shape, y.shape)
  print(y.values)
  avgY = np.average(y.values)
  # exit(0)
  # THE MODEL
  with pm.Model() as model:
    # assumption: runtime = a1 b1k^k n^e1n m^e1m k^e1k + a2 n^e2n m^e2m + a3
    # define hyperpriors
    a1 = pm.Normal('a1', mu = avgY, sd = 10)
    a2 = pm.Normal('a2', mu = avgY, sd = 10)
    a3 = pm.Normal('a3', mu = avgY, sd = 10)
    b1k = pm.Gamma('b1k', mu = 2, sd = 3)
    # b1k = 2
    e1n = pm.Uniform('e1n', 0.5, 3)
    e1m = pm.Uniform('e1m', 0.5, 3)
    e1k = pm.Uniform('e1k', 0.5, 3)
    e2m = pm.Uniform('e2m', 0.5, 3)
    e2n = pm.Uniform('e2n', 0.5, 3)
    eps = pm.Uniform('eps', 0, 6) # infer from data
    # n = x["n"].values
    # m = x["m"].values
    # k = x["k"].values
    n = 5
    m = 4
    k = 2
    # mu = a1 * b1k**k * n**e1n * m**e1m + a2 * n**e2n * m**e2m  + a3
    mu = a1  * n**e1n * m**e1m + a2 * n**e2n * m**e2m  + a3
    # * n**e2n * m**e2m
    # mu = a1 * b1k**k * n**e1n * m**e1m * k**e1k + a2 * n**e2n * m**e2m + a3
    # mu = a1 * b1k**x.k * x.n**e1n * x.m^e1m * x.k^e1k + a2 * x.n^e2n * x.m^e2m + a3
    # tdfB = 1 + tdfBgain * (-pm.math.log(1 - udfB))
    # # define the priors
    # tau = pm.Gamma('tau', 0.01, 0.01)
    # beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    # beta1 = pm.StudentT('beta1', mu=muB, lam=tauB, nu=tdfB, shape=n_predictors)
    # mu = beta0 + pm.math.dot(beta1, x.values.T)
    # define the likelihood
    # mu = beta0 + beta1[0] * x.values[:,0] + beta1[1] * x.values[:,1]
    yl = pm.Normal('yl', mu=mu, tau=eps,) # observed=[40,] * 60) #  y.values)
    # Generate a MCMC chain
    trace = pm.sample(5000, chains=4, cores=1)
  # print(trace["mu"])

  if saveTrace:
    traceFolderName = f"{fileName}_trace"
    if os.path.exists(traceFolderName):
      ind = 0
      while os.path.exists(f"{traceFolderName}_{ind}"):
        ind += 1
      traceFolderName = f"{traceFolderName}_{ind}"
    pm.save_trace(trace, directory=traceFolderName)
    with open(os.path.join(traceFolderName, "pickeledTrace.pkl"), 'wb') as buff:
      pickle.dump({'model': model, 'trace': trace}, buff)
    print(f"{traceFolderName} is saved!")

    pm.traceplot(trace)
    plt.savefig(f"{traceFolderName}/tracePlot.png")
    print(f"{traceFolderName}/tracePlot.png is saved!")

