"""Indexed, pickle-free reading of numeric NPZ arrays for large audits."""
import zipfile
import numpy as np


class NpzArrays:
    def __init__(self,path):
        self.archive=zipfile.ZipFile(path)
        names=self.archive.namelist()
        if len(names)!=len(set(names)) or any(not name.endswith('.npy') for name in names):
            self.archive.close()
            raise ValueError('Expected unique NPY entries in numeric NPZ archive')
        self.files={name[:-4] for name in names}

    def __contains__(self,key):
        return key in self.files

    def __getitem__(self,key):
        if key not in self.files:
            raise KeyError(key)
        with self.archive.open(key+'.npy') as stream:
            return np.lib.format.read_array(stream,allow_pickle=False)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.archive.close()
