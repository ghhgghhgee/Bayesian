from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from scripts._npz_arrays import NpzArrays
ROOT=Path(__file__).resolve().parents[1]


class NpzArrayTests(unittest.TestCase):
    def test_matches_numpy_for_vectors_and_histories(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'arrays.npz'
            values={'x':np.arange(8,dtype=float),'history':np.eye(8),'times':np.arange(8)/10}
            np.savez_compressed(path,**values)
            with NpzArrays(path) as arrays:
                self.assertEqual(arrays.files,set(values))
                for name,value in values.items():
                    self.assertIn(name,arrays)
                    np.testing.assert_array_equal(arrays[name],value)
                with self.assertRaises(KeyError):arrays['absent']

    def test_rejects_duplicate_entries_and_object_arrays(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
            path=Path(folder)/'objects.npz'
            np.savez_compressed(path,object_value=np.array([{'x':1}],dtype=object))
            with NpzArrays(path) as arrays:
                with self.assertRaises(ValueError):arrays['object_value']
            duplicate=Path(folder)/'duplicate.npz'
            with zipfile.ZipFile(duplicate,'w') as archive:
                archive.writestr('x.npy',b'first')
                with self.assertWarns(UserWarning):archive.writestr('x.npy',b'second')
            with self.assertRaises(ValueError):NpzArrays(duplicate)


if __name__=='__main__':unittest.main()
