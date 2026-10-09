import numpy as np
from data.physionet import make_split

def test_make_split_sizes_for_set_a():
    train, val, test = make_split(4000, 0)
    assert (len(train), len(val), len(test)) == (2800, 400, 800)

def test_make_split_small_example():
    train, val, test = make_split(10, 0)
    assert (len(train), len(val), len(test)) == (7, 1, 2)

def test_make_split_uses_every_row_exactly_once():
    for n in [10, 13, 3997, 4000]:
        train, val, test = make_split(n, 0)
        rows = np.concatenate([train, val, test])
        assert sorted(rows.tolist()) == list(range(n))

def test_make_split_same_seed_gives_same_split():
    first = make_split(4000, 7)
    second = make_split(4000, 7)
    for a, b in zip(first, second):
        assert (a == b).all()

def test_make_split_different_seed_gives_different_split():
    train_a, _, _ = make_split(4000, 7)
    train_b, _, _ = make_split(4000, 8)
    assert not (train_a == train_b).all()
