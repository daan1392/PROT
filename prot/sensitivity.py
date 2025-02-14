import pandas as pd
import numpy as np
import serpentTools

MT_translation_dict = {
    'mt 2 xs': (2),
    'mt 4 xs': (4),
    'mt 18 xs': (18),
    'mt 102 xs': (102),
    'nubar prompt': (456),
    'chi prompt': (35018)
}

class Sensitivity(pd.DataFrame):
    """
    Class for handling sensitivity data.
    """
    @property
    def _constructor(self):
        return Sensitivity

    @staticmethod
    def from_sens0(sens0_path: str, 
                   title: str,
                   materiallist=['total'], 
                   zailist=None,
                   pertlist=['mt 2 xs', 'mt 4 xs', 'mt 18 xs', 'mt 102 xs', 'nubar prompt', 'chi prompt']):
        """
        Returns a sensitivity object from a Serpent sensitivity file.

        Parameters
        ----------
        sens0_path : str
            Path to the Serpent sensitivity file.
        title : str
            Title for the sensitivity data.
        materiallist : list, optional
            List of materials to include. Default is ['total'].
        zailist : list, optional
            List of ZAIs to include. Default is None, which includes all ZAIs.
        pertlist : list, optional
            List of perturbations to include. Default is ['mt 2 xs', 'mt 4 xs', 'mt 18 xs', 'mt 102 xs', 'nubar prompt', 'chi prompt'].

        Returns
        -------
        Sensitivity
            Sensitivity object containing the sensitivity data.

        Raises
        ------
        SensitivityError
            If there is an error reading the sensitivity file.
        """
        sens = serpentTools.read(sens0_path)

        materiallist = materiallist or sens.materials
        zailist = zailist or sens.zais
        if 'total' in zailist:
            sens.zais.pop('total')
        pertlist = pertlist or sens.perts

        records = []
        for zai in zailist:
            for pert in pertlist:
                for i in range(len(sens.energies) - 1):
                    record = {
                        'ZAI': zai,
                        'MT': pert,
                        'E_min_eV': np.round(sens.energies[i] * 1e6, decimals=6),
                        'E_max_eV': np.round(sens.energies[i + 1] * 1e6, decimals=6),
                        title: sens.sensitivities["keff"][sens.materials["total"]][sens.zais[zai]][sens.perts[pert]][i, 0],
                        f'{title}_std' : sens.sensitivities["keff"][sens.materials["total"]][sens.zais[zai]][sens.perts[pert]][i, 1]
                    }
                    records.append(record)

        df = pd.DataFrame(records)
        df.set_index(['ZAI', 'MT', 'E_min_eV', 'E_max_eV'], inplace=True)
        df.index = df.index.set_levels(
            df.index.levels[1].map(MT_translation_dict), level='MT'
        )
        return Sensitivity(df)
    
def sandwich(s1, cov, s2):
    """
    Perform the sandwich operation for sensitivity analysis.

    Parameters
    ----------
    s1 : pd.Series
        First sensitivity vector.
    cov : pd.DataFrame
        Covariance matrix.
    s2 : pd.Series
        Second sensitivity vector.

    Returns
    -------
    float
        Result of the sandwich operation.
    """
    idx = s1.index.intersection(cov.index)
    return s1.loc[idx].T @ cov.loc[idx, idx] @ s2.loc[idx]