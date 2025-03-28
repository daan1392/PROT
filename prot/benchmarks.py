import serpentTools
import pandas as pd
import numpy as np
from .sensitivity import Sensitivity, sandwich
import h5py
from .nuclear_data import Covariances
from sandy import zam2latex

class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, 
                 results_path=None,
                 sens_path=None, res_path=None):
        """
        Initialize the Benchmark class.

        Parameters
        ----------
        sens_path : str, optional
            Path to the sensitivity file. Only needed when not loading from h5.
        res_path : str, optional
            Path to the results file. Only needed when not loading from h5.
        """
        self.sens_path = sens_path
        self.res_path = res_path
        self.results_path = results_path
        self.title = None
        self.S = None
        self.K_exp = None  
        self.K_exp_std = None  
        self.K_calc = None
        self.K_calc_std = None

        if sens_path is not None and res_path is not None:
            self.set_simulation_results()
            self.set_sensitivity()
            self.set_experimental_results()
            self.zais = self.S.index.get_level_values('ZAI').unique()

    def set_sensitivity(self):
        """
        Read sensitivities.
        """
        self.S = Sensitivity(Sensitivity.from_sens0(self.sens_path, self.title))

    def set_simulation_results(self):
        """
        Retrieve information from the results file.
        """
        res = serpentTools.read(self.res_path)
        self.K_calc = res.resdata['absKeff'][0] 
        self.K_calc_std = res.resdata['absKeff'][1]
        self.title = res.metadata['inputFileName'].split('.')[0]

    def print_summary(self):
        """
        Print a summary of the most important reactions and discrepancy.
        """
        print(f"Title: {self.title}\n Bias={self.K_calc-self.K_exp:.5f} +/- {np.sqrt(self.K_calc_std**2+self.K_exp_std**2):.5f}")

    def set_experimental_results(self):
        """
        Read experimental results from ICSBEP database.

        Parameters
        ----------
        path : str, optional
            Path to the experimental results file. Default is 'C:\\Users\\dhouben\\Documents\\Benchmarks\\exp_results.xlsx'.
        """
        df = pd.read_excel(self.results_path, index_col='title')
        self.K_exp = df.loc[self.title]['E. Mean']
        self.K_exp_std = df.loc[self.title]['E. Std']
        
    def to_hdf5(self, file_path='benchmarks.h5'):
        """
        Save the benchmark data to an HDF5 file.

        Parameters
        ----------
        file_path : str, optional
            The path to the HDF5 file where the benchmark data will be saved. Default is 'benchmarks.h5'.
        """
        with h5py.File(file_path, 'a') as f:
            # Create group for this benchmark if it exists
            if self.title in f:
                del f[self.title]

            # Create group and save attributes
            grp = f.create_group(self.title)
            grp.attrs['K_exp'] = self.K_exp
            grp.attrs['K_exp_std'] = self.K_exp_std
            grp.attrs['K_calc'] = self.K_calc
            grp.attrs['K_calc_std'] = self.K_calc_std

        # Save sensitivity DataFrame using pandas to_hdf
        self.S.to_hdf(path_or_buf=file_path, key=f'{self.title}/sensitivity', mode='a', format='table')

    @classmethod
    def from_hdf5(cls, file_path, title):
        """
        Create a Benchmark instance from an HDF5 file.

        Parameters
        ----------
        file_path : str
            Path to the HDF5 file containing the benchmark data.
        title : str
            Title of the benchmark to load.

        Returns
        -------
        Benchmark
            A new Benchmark instance with the loaded data.
        """
        benchmark = cls()
    
        with h5py.File(file_path, 'r') as f:
            if title not in f:
                raise KeyError(f"Benchmark '{title}' not found in {file_path}")
            
            grp = f[title]
            benchmark.title = title
            benchmark.K_exp = grp.attrs['K_exp']
            benchmark.K_exp_std = grp.attrs['K_exp_std']
            benchmark.K_calc = grp.attrs['K_calc']
            benchmark.K_calc_std = grp.attrs['K_calc_std']

        # Load sensitivity DataFrame using pandas read_hdf
        benchmark.S = pd.read_hdf(file_path, f'{title}/sensitivity')

        return benchmark
    
    @classmethod
    def from_sens_res(cls, sens_path, res_path):
        """
        Create a Benchmark instance from sensitivity and results files.

        Parameters
        ----------
        sens_path : str
            Path to the sensitivity file.
        res_path : str
            Path to the results file.

        Returns
        -------
        Benchmark
            A new Benchmark instance with the loaded data.
        """
        benchmark = cls(sens_path, res_path)
        return benchmark
    
    # def get_nd_std_decomposed(self, covariances):

class BenchmarkSuite:
    """Class for running a suite of benchmarks."""

    def __init__(self, benchmarks: list = None):
        """
        Initialize the BenchmarkSuite class.

        Parameters
        ----------
        benchmarks : list, optional
            List of Benchmark objects. Default is None.
        """
        self.benchmarks = benchmarks if benchmarks is not None else []
        self.V_prior = None
        self.C_inv = None
        self.update_suite()

    def update_suite(self):
        """
        Update the suite dataframes and matrices.
        """
        self.S = pd.concat([benchmark.S.iloc[:,0].to_frame() for benchmark in self.benchmarks], axis=1).fillna(0)
        self.titles = [benchmark.title for benchmark in self.benchmarks]
        self.ZAIs = self.S.index.get_level_values('ZAI').unique()
        self.K_exp = pd.DataFrame({'K_exp': [benchmark.K_exp for benchmark in self.benchmarks]}, 
                       index=self.titles)
        self.V_exp = pd.DataFrame(np.diag([benchmark.K_exp_std**2 for benchmark in self.benchmarks]), index=self.titles, columns=self.titles)
        self.K_prior = pd.DataFrame({'K_prior': [benchmark.K_calc for benchmark in self.benchmarks]}, 
                       index=self.titles)
        
    def from_hdf5(file_path=None, titles=None):
        if not titles:
            with h5py.File(file_path, 'r') as f:
                titles = list(f.keys())
            
        return BenchmarkSuite([Benchmark.from_hdf5(file_path, title) for title in titles])

    def get_benchmark(self, title):
        """
        Get a benchmark from the suite.

        Parameters
        ----------
        title : str
            Title of the benchmark to be retrieved.

        Returns
        -------
        Benchmark
            Benchmark object.
        """
        return [benchmark for benchmark in self.benchmarks if benchmark.title == title][0]
        
    def remove_benchmark(self, title):
        """
        Remove a benchmark from the suite.

        Parameters
        ----------
        title : str
            Title of the benchmark to be removed.
        """
        self.benchmarks = [benchmark for benchmark in self.benchmarks if benchmark.title != title]
        self.update_suite()

    def add_benchmark(self, benchmark):
        """
        Add a benchmark to the suite.

        Parameters
        ----------
        benchmark : Benchmark
            Benchmark object to be added.
        """
        self.benchmarks.append(benchmark)
        self.update_suite()

    def get_titles(self):
        """
        Get the titles of all benchmarks in the suite.

        Returns
        -------
        list
            List of benchmark titles.
        """
        return [benchmark.title for benchmark in self.benchmarks]

    def calculate_V_prior(self, covariances):
        """
        Calculate the prior output uncertainty due to nuclear data using first-order Taylor approximation of the sensitivity coefficient.

        Parameters
        ----------
        covariances : dict
            Dictionary of covariance matrices.
        """
        self.V_prior = pd.concat({
            zai: sandwich(g.reset_index(level="ZAI", drop=True), covariances[zai], g.reset_index(level="ZAI", drop=True)) for zai, g in self.S.groupby("ZAI")
        }).groupby(level=1).sum()

    def get_nd_std(self):
        """
        Return a pandas.Series object where the index is the benchmark name,
        the value is the standard deviation coming from the nuclear data.
        
        Returns
        -------
        pd.Series
            Nuclear data standard deviation.
        """
        if not self.V_prior.empty:
            return pd.Series(np.sqrt(np.diag(self.V_prior)), index=self.V_prior.index).rename("Prior ND std")
        else:
            # Raise warning
            print("calculate_V_prior(covariance) not yet performed")
            return None


    def calculate_C_inv(self, covariances=None):
        """
        Calculate the inverse of the covariance matrix.

        Parameters
        ----------
        covariances : dict, optional
            Dictionary of covariance matrices. Default is None.

        Returns
        -------
        pd.DataFrame
            Inverse of the covariance matrix.
        """
        if self.V_prior is None:
            raise ValueError('V_prior not yet calculated, provide covariances first')
        
        if covariances is None:
            self.C_inv = pd.DataFrame(np.linalg.pinv(self.V_prior + np.diag(self.V_exp)), self.V_prior.columns, self.V_prior.index)
            return self.C_inv
        else:
            V = pd.concat({
                zai: sandwich(g.reset_index(level="ZAI", drop=True), covariances[zai], g.reset_index(level="ZAI", drop=True)) for zai, g in self.S.groupby("ZAI")
            }).groupby(level=1).sum()
            C_inv = pd.DataFrame(np.linalg.pinv(V + np.diag(self.V_exp)), V.columns, V.index)
            return C_inv

    def calculate_reduced_chi(self, K=None, covariances=None):
        """
        Calculate the reduced chi-squared value.

        Parameters
        ----------
        K : pd.Series, optional
            Series of keff values. Default is None.
        covariances : dict, optional
            Dictionary of covariance matrices. Default is None.

        Returns
        -------
        float
            Reduced chi-squared value.
        """
        K_exp = self.K_exp['K_exp']

        K = self.K_prior['K_prior'] if K is None else K

        C_inv = self.calculate_C_inv() if covariances is None else self.calculate_C_inv(covariances)
            
        X = (K - K_exp).T @ C_inv @ (K - K_exp) / C_inv.shape[0]
        return X
    
    def gls(self, covariances):
        """
        Perform Generalized Least Squares (GLS) adjustment.

        Parameters
        ----------
        covariances : dict
            Dictionary of covariance matrices.

        Returns
        -------
        dict
            Dictionary containing the results of the GLS adjustment.
        """

        V_prior_y_N = {}
        for zai in self.ZAIs:
            V_prior_y_N[zai] = sandwich(self.S.loc[zai], covariances[zai], self.S.loc[zai])
            V_prior_y_N[zai].columns = V_prior_y_N[zai].index
        V_prior_y = pd.concat(V_prior_y_N).groupby(level=1).sum() 
        
        self.C_inv = pd.DataFrame(np.linalg.pinv(V_prior_y + self.V_exp), V_prior_y.columns, V_prior_y.index)
        A_N = {}
        delta_xs_N = {}
        del_K_N = {}
        V_post_N = {}
        V_post_K_N = {}

        K_diff = (self.K_exp['K_exp'] - self.K_prior['K_prior']) / self.K_prior['K_prior']

        for zai in self.ZAIs:
            S_zai = self.S.loc[zai]
            idx = covariances[zai].index.intersection(S_zai.index)
            S_zai_idx = S_zai.loc[idx]

            A_N[zai] = covariances[zai].loc[idx, idx] @ S_zai_idx

            # Nuclear data
            delta_xs_N[zai] = A_N[zai] @ self.C_inv @ K_diff
            V_post_N[zai] = covariances[zai].loc[idx, idx] - A_N[zai] @ self.C_inv @ A_N[zai].T

            # Output
            del_K_N[zai] = S_zai_idx.T @ delta_xs_N[zai]
            V_post_K_N[zai] = S_zai_idx.T @ V_post_N[zai].loc[idx, idx] @ S_zai_idx
            
        del_K_post = pd.concat(del_K_N, axis=1).sum(axis=1)
        K_post = self.K_prior['K_prior'] + del_K_post
        V_post_K = pd.concat(V_post_K_N).groupby(level=1).sum()

        res = {
            'Bmarks incl.': self.get_titles,
            'prior X'     : self.calculate_reduced_chi(),
            'post X'      : self.calculate_reduced_chi(K_post, V_post_N),
            'delta_xs'    : delta_xs_N,
            'V_post_N'    : V_post_N,
            'K_post'      : K_post,
            'del_K_post'  : del_K_post,
            'V_post_K'    : V_post_K
        }

        print(f"GLS Procedure Summary:")
        print(f"Prior X: {res['prior X']:.3f}")
        print(f"Posterior X: {res['post X']:.3f}")
        print(f"Titles of Benchmarks Included: {', '.join(res['Bmarks incl.']())}")
        print(f"Condition Number: {np.linalg.cond(self.C_inv.values):.2f}")
        
        return res
    
    def get_covariances(self, nd_path):
        """
        Load covariance matrices from a nuclear data file.

        Parameters
        ----------
        nd_path : str
            Path to the nuclear data file.

        Returns
        -------
        dict
            Dictionary of covariance matrices.
        """
        covariances = {}
        for zai in self.ZAIs:
            covariances[zai] = Covariances.from_hdf5(nd_path, zai).fillna(0)
            if covariances[zai].empty:
                covariances[zai] = pd.DataFrame(0, index=[zai], columns=[zai])