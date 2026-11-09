import numpy as np
import pandas as pd

from foodcomp.taiwan_deduplicated_v1 import (
    TEST_PROFILES,
    coverage_nested_order,
    group_aware_split,
)


def test_group_aware_split_keeps_identity_groups_intact():
    rows = np.arange(700, dtype=np.int64)
    keys = [f"pair:{index // 2}" if index < 40 else f"single:{index}" for index in rows]
    clean = pd.DataFrame({"profile_index": rows, "identity_group_key": keys})
    test, adaptation = group_aware_split(clean)
    assert len(test) == TEST_PROFILES
    assert len(adaptation) == len(clean) - TEST_PROFILES
    by_row = clean.set_index("profile_index").identity_group_key
    assert not (set(by_row.loc[test]) & set(by_row.loc[adaptation]))


def test_coverage_order_is_nested_and_uses_missingness_only():
    pool = np.arange(20, dtype=np.int64)
    observed = np.zeros((20, 5), dtype=bool)
    observed[0, [1, 2, 3]] = True
    observed[1:, 1] = True
    order, coverage = coverage_nested_order(
        pool, np.asarray([1, 2, 3]), observed, seed=17
    )
    assert coverage == 1
    assert order[0] == 0
    assert set(order[:8]) <= set(order[:32])
    assert sorted(order.tolist()) == pool.tolist()
