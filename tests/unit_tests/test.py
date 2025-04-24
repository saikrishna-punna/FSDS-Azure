from unittest import TestCase

import numpy as np
import pandas as pd

from ingest_data import load_data
from score import scorer
from train import training


class TestTrain(TestCase):
    def test_two(self):
        f = training()
        model = not (len(dir(f)) == len(dir(type(f)())))
        self.assertTrue(model, "not a model")


class TestIngest(TestCase):
    def test_one(self):
        a, b, c, d = load_data()
        assert isinstance(a, pd.DataFrame)
        assert isinstance(b, pd.Series)
        assert isinstance(c, pd.DataFrame)
        assert isinstance(d, pd.Series)


class TestScorer(TestCase):
    def test_three(self):
        a = scorer()
        assert type(a) is np.float64, "num must be an floating integer"

