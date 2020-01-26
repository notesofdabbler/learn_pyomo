import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.sparse import BlockSymMatrix, BlockMatrix, BlockVector
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

def getvarorder(nlp, parm_vars, non_parm_vars):
    # ensure that variable order from original hessian is preserved
    varnames = [x.name for x in nlp.get_pyomo_variables()]
    #print(varnames)

    curr_order = {k.name:i for (i,k) in enumerate(non_parm_vars)}
    parm_order = {k.name:(i + len(curr_order)) for (i,k) in enumerate(parm_vars)}
    curr_order.update(parm_order)
    #print(curr_order)

    zorder = [curr_order[k] for k in varnames]
    #print('zorder:', zorder)
    
    return zorder

def getZ(nlp, parm_vars):
    # Get the Z matrix to compute reduced hessian
    parm_vars_name = [x.name for x in parm_vars]
    non_parm_vars = [x for x in nlp.get_pyomo_variables() if x.name not in parm_vars_name]
    
    Ji = nlp.extract_submatrix_jacobian(pyomo_variables = parm_vars, pyomo_constraints=nlp.get_pyomo_constraints())
    Jd = nlp.extract_submatrix_jacobian(pyomo_variables = non_parm_vars, pyomo_constraints=nlp.get_pyomo_constraints())
    #print("Ji")
    #print(Ji.todense())
    #print("Jd")
    #print(Jd.todense())

    Zd = spsolve(Jd.tocsc(), Ji.tocsc())
    Z = BlockMatrix(2, 1)
    Z[0, 0] = Zd
    Z[1, 0] = identity(len(parm_vars))
    #print("Z")
    #print(Z.todense())
    
    # reorder variables to the order in hessian
    zorder = getvarorder(nlp, parm_vars, non_parm_vars)
    Zorder = Z.tocsc()[zorder, :].todense()
    #print("Zorder")
    #print(Zorder)
    
    return Zorder

def getHred(nlp, parm_vars):
    # compute reduced hessian
    H = nlp.evaluate_hessian_lag()
    if nlp.n_primals() == len(parm_vars):
        Hred = H.todense()
    else:
        Zorder = getZ(nlp, parm_vars)
        Hred = Zorder.T.dot(H.dot(Zorder))
    return Hred