import pandas as pd
import numpy as np
import scipy.sparse

class CrossSections(pd.DataFrame):
    """Class for handling cross-section data."""
    
    @property
    def _constructor(self):
        return CrossSections

    def to_hdf5(self, file_path: str, zai: int, temperature: float, err: float):
        """Save CrossSections data to HDF5 file."""
        zai = str(zai)
        with pd.HDFStore(file_path, mode='a') as store:
            # Save DataFrame
            store.put(
                f'zai_{zai}/cross_sections',
                self
            )
            store.get_storer(f'zai_{zai}/cross_sections').attrs.temperature = temperature
            store.get_storer(f'zai_{zai}/cross_sections').attrs.err = err

    @staticmethod
    def from_hdf5(file_path: str, zai: int, mts=None):
        """Load CrossSections data from HDF5 file."""
        zai = str(zai)
        with pd.HDFStore(file_path) as store:
            # Check if the zai exists in the file
            if f'zai_{zai}/cross_sections' not in store:
                return CrossSections()  # Return an empty CrossSections object
            
            # Accessing the xs data
            xs = store[f'zai_{zai}/cross_sections']

            if xs.empty:
                return CrossSections(xs)
            elif mts:
                mts_av = [mt for mt in mts if mt in xs.index.get_level_values('MT').unique()]
                return CrossSections(xs.loc[mts_av])
            else:
                return CrossSections(xs)

class Covariances(pd.DataFrame):
    """Class for handling covariance data."""
    
    @property
    def _constructor(self):
        return Covariances

    def to_hdf5(self, file_path: str, zai: int, temperature: float, err: float):
        """Save Covariances data using pandas sparse format, storing only lower triangular."""
        zai = str(zai)
        if not isinstance(self.index, pd.MultiIndex):
            raise ValueError(f'Covariances not found for {zai}.')
        
        # Get lower triangular indices
        rows, cols = np.tril_indices(len(self))
        
        # Create sparse data for lower triangular
        data = self.values[rows, cols]
        
        # Create DataFrame with sparse data
        sparse_data = pd.DataFrame({
            'row': rows,
            'col': cols,
            'value': data
        })
        
        # Filter out zeros to save space
        sparse_data = sparse_data[sparse_data['value'] != 0]
        
        with pd.HDFStore(file_path, mode='a') as store:
            # Save index structure separately to handle MultiIndex
            if isinstance(self.index, pd.MultiIndex):
                for i, name in enumerate(self.index.names):
                    store.put(
                        f'zai_{zai}/covariances_index/level_{i}',
                        pd.Series(self.index.get_level_values(i)),
                        format='table'
                    )
            else:
                store.put(
                    f'zai_{zai}/covariances_index/values',
                    pd.Series(self.index),
                    format='table'
                )
            
            # Save sparse data
            store.put(
                f'zai_{zai}/covariances/data',
                sparse_data,
                format='table',
                complevel=9,
                complib='blosc'
            )
            
            # Save metadata
            attrs = store.get_storer(f'zai_{zai}/covariances/data').attrs
            attrs.shape = self.shape
            attrs.index_names = self.index.names
            attrs.is_multiindex = isinstance(self.index, pd.MultiIndex)
            attrs.temperature = temperature
            attrs.err = err

    @classmethod
    def from_hdf5(cls, file_path: str, zai: int, mts=None):
        """Load sparse Covariances data from HDF5."""
        zai = str(zai)
        
        with pd.HDFStore(file_path, mode='r') as store:
            # Load sparse data
            if f'zai_{zai}/covariances/data' in store:
                sparse_data = store[f'zai_{zai}/covariances/data']
            else:
                print(f"No covariance data found for {zai}, returning empty df")
                
                return pd.DataFrame()
            
            attrs = store.get_storer(f'zai_{zai}/covariances/data').attrs
            
            # Reconstruct index (will be used for both rows and columns)
            if attrs.is_multiindex:
                index_levels = [
                    store[f'zai_{zai}/covariances_index/level_{i}']
                    for i in range(len(attrs.index_names))
                ]
                index = pd.MultiIndex.from_arrays(
                    index_levels,
                    names=attrs.index_names
                )
            else:
                index = store[f'zai_{zai}/covariances_index/values']
            
            # Filter by MTs if needed
            if mts is not None:
                mask = index.get_level_values('MT').isin(mts)
                index = index[mask]
                
                # Update sparse data to include only relevant rows/columns
                idx_map = {old: new for new, old in enumerate(np.where(mask)[0])}
                sparse_data = sparse_data[
                    sparse_data['row'].isin(idx_map.keys()) & 
                    sparse_data['col'].isin(idx_map.keys())
                ].copy()
                sparse_data['row'] = sparse_data['row'].map(idx_map)
                sparse_data['col'] = sparse_data['col'].map(idx_map)
                shape = (len(index), len(index))
            else:
                shape = attrs.shape
            
            # Create sparse matrix using scipy.sparse for memory efficiency
            sparse_matrix = scipy.sparse.coo_matrix(
                (sparse_data['value'], 
                 (sparse_data['row'], sparse_data['col'])),
                shape=shape
            )
            
            # Convert to symmetric matrix efficiently
            full_matrix = sparse_matrix + sparse_matrix.T
            # Subtract diagonal to avoid doubling it
            full_matrix.setdiag(sparse_matrix.diagonal())
            
            # Return plain DataFrame instead of Covariances instance
            return pd.DataFrame(
                full_matrix.toarray(), 
                index=index, 
                columns=index
            )