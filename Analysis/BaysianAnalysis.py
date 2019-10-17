from Utils.const import *

import numpy as np
import pymc3 as pm
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import theano.tensor as tt

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
  stdY = np.std(y.values, ddof = 1)

  b1kMu = 2
  av = avgY / ((b1kMu) ** 3)
  # a1, a2, a3 = av, av, av
  # e1n, e1m, e1k, e2m, e2n = 1, 1, 1, 1, 1
  # eps = stdY / 10
  n = x["n"].values[0]
  m = x["m"].values[0]
  k = x["k"].values[0]

  # exit(0)
  # THE MODEL
  with pm.Model() as model:
    # assumption: runtime = a1 b1k^k n^e1n m^e1m k^e1k + a2 n^e2n m^e2m + a3
    # define hyperpriors
    a1 = pm.Normal('a1', mu = av, sd = 1)
    a2 = pm.Normal('a2', mu = av, sd = 1)
    a3 = pm.Normal('a3', mu = av, sd = 1)
    b1k = pm.Gamma('b1k', mu = b1kMu, sd = 3)
    e1n = pm.Uniform('e1n', 0.1, 2)
    e1m = pm.Uniform('e1m', 0.1, 2)
    e1k = pm.Uniform('e1k', 0.1, 2)
    e2m = pm.Uniform('e2m', 0.1, 2)
    e2n = pm.Uniform('e2n', 0.1, 2)
    eps = pm.Uniform('eps', 0.8, 1) # infer from data

    # Cunstruct input:

    # print(a1, a2, a3, b1k, e1n, e1m, e1k, e2m , e2n, eps, n, m, k)
    # n = 1
    # m = 1
    # k = 1
    # mu = a1 * b1k**k * n**e1n * m**e1m + a2 * n**e2n * m**e2m  + a3
    # mu = a1  * n**e1n * m**e1m + a2 * n**e2n * m**e2m  + a3
    # * n**e2n * m**e2m
    # mu = tt.as_tensor_variable(a1 * b1k**k * n**e1n * m**e1m * k**e1k + a2 * n**e2n * m**e2m + a3)
    mu = pm.Deterministic("mu", a1 * b1k**k * n**e1n * m**e1m * k**e1k + a2 * n**e2n * m**e2m + a3)
    # mu = a1 * b1k**x.k * x.n**e1n * x.m^e1m * x.k^e1k + a2 * x.n^e2n * x.m^e2m + a3
    # tdfB = 1 + tdfBgain * (-pm.math.log(1 - udfB))
    # # define the priors
    # tau = pm.Gamma('tau', 0.01, 0.01)
    # beta0 = pm.Normal('beta0', mu=0, tau=1.0E-12)
    # beta1 = pm.StudentT('beta1', mu=muB, lam=tauB, nu=tdfB, shape=n_predictors)
    # mu = beta0 + pm.math.dot(beta1, x.values.T)
    # define the likelihood
    # mu = beta0 + beta1[0] * x.values[:,0] + beta1[1] * x.values[:,1]
    # yl = pm.Deterministic("yl", mu)
    yl = pm.Normal('yl', mu=mu, sigma=eps, observed=  y.values)
    # Generate a MCMC chain
    trace = pm.sample(draws = 1000, chains=4, cores=1, tune = 1000)
  # print(trace["mu"])
  # print(type(trace["yl"]))
  # print(trace["yl"].shape)
  # madeUpDf = pd.DataFrame()
  # madeUpDf["runtime"]=trace["yl"].reshape(-1)
  # madeUpDf["runtime2"]=trace["yln"].reshape(-1)
  # madeUpDf["n"] = [n,] * madeUpDf.shape[0]
  # madeUpDf["m"] = [m,] * madeUpDf.shape[0]
  # madeUpDf["k"] = [k,] * madeUpDf.shape[0]

  # print(madeUpDf)

  # madeUpDf.to_csv("madeUpInput.csv")
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

