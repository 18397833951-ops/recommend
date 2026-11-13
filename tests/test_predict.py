import csv
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from predict import write_predictions


class ExportTest(unittest.TestCase):
    def test_preserves_original_fields_order_and_continuous_scores(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as folder:
            path=Path(folder)/'predictions.csv'
            rows=[['12','30','400'],['2','8','100']]
            scores=np.array([4.123456789123,2.987654321098])
            write_predictions(rows,scores,path)
            with path.open(encoding='utf-8-sig',newline='') as stream:
                saved=list(csv.reader(stream))
            self.assertEqual(saved[0],['userId','movieId','timestamp','prediction'])
            self.assertEqual([r[:3] for r in saved[1:]],rows)
            for row, score in zip(saved[1:],scores):
                self.assertAlmostEqual(float(row[3]),score,places=9)
            with self.assertRaises(ValueError):
                write_predictions(rows,np.array([float('nan'),2.]),path)
            with self.assertRaises(ValueError):
                write_predictions(rows,np.array([2.]),path)


if __name__=='__main__':
    unittest.main()
