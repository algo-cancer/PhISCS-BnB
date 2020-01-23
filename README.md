# PhISCS-BnB

PhISCS-BnB is a tool 

## Contents
  1. [Installation](#installation)
     * [PhISCS-I](#installationilp)
       * [Prerequisite: ILP solver](#prerequisiteilp)
     * [PhISCS-B](#installationcsp)
       * [Prerequisite: CSP solver](#prerequisitecsp)
  2. [Running](#running)
     * [Input](#input)
     * [Output](#output)
       * [Log File](#logfile)
       * [Output Matrix File](#outputmatrixfile)
     * [Parameters](#parameters)
  3. [Example](#example)
  4. [Contact](#contact)

<a name="installation"></a>
## Installation
PhISCS is written in Python and C. It supports both Python 2.7 and 3. Currently it is intended to be run on POSIX-based systems (only Linux and macOS have been tested).

> **RECOMENDATION**:  At the moment, in cases when both, single-cell and bulk data are used as input, we recommend the use of PhISCS-I over PhISCS-B (due to more thorough tests and software validation that we have performed for PhISCS-I). However, when single-cell data is the only input, we have extensively tested both implementations and, since PhISCS-B can have potential running time advantage in this case, we recommend its use over PhISCS-I.   

<a name="installationilp"></a>
### PhISCS-I
```
git clone --recursive https://github.com/sfu-compbio/PhISCS.git
cd PhISCS
python PhISCS-I --help
```

<a name="prerequisiteilp"></a>
#### Prerequisite: ILP solver

In order to run PhISCS-I, the main requirement is the installation of Gurobi solver. [Gurobi](http://www.gurobi.com) a commercial solver which is free for academic purposes. After installing it, installation of `gurobipy` package is necessary prior to being able to successfully run PhISCS-I (below we provide some examples of the input and commands used to run the tool).


<a name="installationcsp"></a>
### PhISCS-B

```
git clone --recursive https://github.com/sfu-compbio/PhISCS.git
cd PhISCS
./PhISCS-B-configure
python PhISCS-B --help
```

<a name="prerequisitecsp"></a>
#### Prerequisite: CSP solver

Some of CSP solver have been already included in the PhISCS package. There is an option to add a new CSP solver to PhISCS-B by provinding a path to the exe file of the desired CSP solver.


<a name="running"></a>
## Running

<a name="input"></a>
### Input

Single-cell input is assumed to be represented in the form of ternary, __tab-delimited__, matrix with rows corresponding to single-cells and columns corresponding to mutations. We assume that this file contains headers and that matrix is ternary matrix with 0 denoting the absence and 1 denoting the presence of mutation in a given cell, whereas ? represents the lack of information about presence/absence of mutation in a given cell (i.e. missing entry). __In order to simplify parsing of the matrix, we also assume that upper left corner equals to string `cellID/mutID`__.

Below is an example of single-cell data matrix. Note that mutation and cell names are arbitrary strings not containing tabs or spaces, however they must be unique.
```
cellID/mutID  mut0  mut1  mut2  mut3  mut4  mut5  mut6  mut7
cell0         0     0     ?     0     0     0     0     0
cell1         0     ?     1     0     0     0     1     1
cell2         0     0     1     0     0     0     1     1
cell3         1     1     0     0     0     0     0     0
cell4         0     0     1     0     0     0     0     0
cell5         1     0     0     0     0     0     0     0
cell6         0     0     1     0     0     0     1     1
cell7         0     0     1     0     0     0     0     0
cell8         ?     0     0     0     ?     0     ?     1
cell9         0     1     0     0     0     0     0     0
```

<a name="output"></a>
### Output
The program will generate two files in **OUT_DIR** folder (which is set by argument -o or --outDir). This folder will be created automatically if it does not exist.

<a name="outputmatrixfile"></a>
#### 1. Output Matrix File
The output matrix is also a tab-delimited file having the same format as the input matrix, except that eliminated mutations (columns) are excluded (so, in case when mutation elimination is allowed, this matrix typically contains less columns than the input matrix). Output matrix represents genotypes-corrected matrix (where false positives and false negatives from the input are corrected and each of the missing entries set to 0 or 1). Suppose the input file is **INPUT_MATRIX.ext**, the output matrix will be stored in file **OUT_DIR/INPUT_MATRIX.CFMatrix**. For example:
```
 input file: data/ALL2.SC
output file: OUT_DIR/ALL2.CFMatrix
```

<a name="logfile"></a>
#### 2. Log File
Log file contains various information about the particular run of PhISCS (e.g. eliminated mutations or likelihood value). The interpretation of the relevant reported entries in this file is self-evident. Suppose the input file is **INPUT_MATRIX.ext**, the log will be stored in file **OUT_DIR/INPUT_MATRIX.log**. For example:
```
input file: data/ALL2.SC
  log file: OUT_DIR/ALL2.log
```

<a name="parameters"></a>
### Parameters
| Parameter  | Description                                                                                | Default  | Mandatory      |
|------------|--------------------------------------------------------------------------------------------|----------|----------------|
| -SCFile    | Path to single-cell data matrix file                                                       | -        | :radio_button: |
| -fn        | Probablity of false negative                                                               | -        | :radio_button: |
| -fp        | Probablity of false positive                                                               | -        | :radio_button: |
| -o         | Output directory                                                                           | current  | :white_circle: |
| -kmax      | Max number of mutations to be eliminated                                                   | 0        | :white_circle: |
| -threads   | Number of threads (supported by PhISCS-I)                                                  | 1        | :white_circle: |
| -bulkFile  | Path to bulk data file                                                                     | -        | :white_circle: |
| -delta     | Delta parameter accounting for VAF variance                                                | 0.20     | :white_circle: |
| -time      | Max time (in seconds) allowed for the computation                                          | 24 hours | :white_circle: |
| --drawTree | Draw output tree with Graphviz                                                             | -        | :white_circle: |

<a name="example"></a>
## Example

For running PhISCS without VAFs information and without ISA violations:
```
python PhISCS-I -SCFile example/input.SC -fn 0.2 -fp 0.0001 -o result/
```

For running PhISCS without VAFs information but with ISA violations:
```
python PhISCS-I -SCFile example/input.SC -fn 0.2 -fp 0.0001 -o result/ -kmax 1
```

For running PhISCS with both VAFs information and ISA violations (with time limit of 24 hours):
```
python PhISCS-I -SCFile example/input.SC -fn 0.2 -fp 0.0001 -o result/ -kmax 1 -bulkFile example/input.bulk -time 86400
```

For running PhISCS with VAFs information but no ISA violations (with drawing the output tree):
```
python PhISCS-I -SCFile example/input.SC -fn 0.2 -fp 0.0001 -o result/ -bulkFile example/input.bulk --drawTree
```

<a name="contact"></a>
## Contact
If you have any questions please e-mail us at esadeqia@iu.edu or frashidi@iu.edu.
