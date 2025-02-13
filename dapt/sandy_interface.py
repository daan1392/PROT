import sandy
import pandas as pd
import numpy as np
from .nuclear_data import CrossSections, Covariances


def xs_to_df(xs):
    E_left  = xs.index.get_level_values('E').left
    E_right = xs.index.get_level_values('E').right
    new_index = pd.MultiIndex.from_arrays(
        [
            E_left,
            E_right
        ],
        names=["E_min_eV", "E_max_eV"]
    )
    xs.index = new_index
    
    xs = xs.T.droplevel('MAT').T
    return xs.melt(var_name="MT", value_name="Value", ignore_index=False).set_index("MT", append=True).swaplevel(1, 2).swaplevel(0,1)


def cov_to_df(cov, key=None):
    '''
    Extract covariance data for a specific nuclide given a list of MT numbers
    '''
    # Change energy index and index name
    E_left  = cov.index.get_level_values('E').left
    E_right = cov.index.get_level_values('E').right
    new_index = pd.MultiIndex.from_arrays(
        [
            cov.index.get_level_values("MAT"),
            cov.index.get_level_values("MT") + 35000 if key == 'errorr35' else cov.index.get_level_values("MT"),
            E_left,
            E_right
        ],
        names=["MAT", "MT", "E_min_eV", "E_max_eV"]
    )
    cov.index = new_index
    
    cov = cov.T

    E_left  = cov.index.get_level_values('E').left
    E_right = cov.index.get_level_values('E').right
    new_index = pd.MultiIndex.from_arrays(
        [
            cov.index.get_level_values("MAT"),
            cov.index.get_level_values("MT") + 35000 if key == 'errorr35' else cov.index.get_level_values("MT"), # change mt number since it is the same as in MF=3 MT=18
            E_left,
            E_right
        ],
        names=["MAT", "MT", "E_min_eV", "E_max_eV"]
    )
    cov.index = new_index

    cov = cov.droplevel('MAT')
    cov = cov.T
    cov = cov.droplevel('MAT')
    
    if key == 'errorr35':
        # make covariance matrix symmetric
        cov = pd.DataFrame(np.tril(cov.values)+np.tril(cov.values).T - np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
    
    return cov