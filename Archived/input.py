from Utils.const import *
from Utils.interfaces import *
from Utils.util import *
from Boundings.LP import *
from Boundings.MWM import *
from Boundings.CSP import *
from Boundings.LP_APX import *
from Boundings.Hybrid import *
from general_BnB import *
from Boundings.two_sat import *

file_names_list = [
    # "simNo_2-s_4-m_50-n_50-k_50.SC.noisy",
    # "SCTrioSeq_cancer_genes.SC",
    # "SCTrioSeq_cancer_genes_LN.SC",
    "Chi-Ping.SC",
]

methods = [
    # (PhISCS_I, None),
    # (PhISCS_B, None),
    # ('BnB_0', two_sat(priority_version=-1)),
    # ('BnB_1', two_sat(priority_version=-1)),
    # ('BnB_1', two_sat(priority_version=-5)),
    ('BnB_1', two_sat(priority_version=-1, formulation_version=0, formulation_threshold=0)),
]
