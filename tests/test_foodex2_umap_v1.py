import pandas as pd

from foodcomp.foodex2_umap_v1 import (
    FISH, GRAINS, LEGUMES, MEAT, VEGETABLES,
    assign_foodex2_labels, direct_foodex2_label,
)


def test_source_code_crosswalks_are_conservative():
    assert direct_foodex2_label("mext_japan_2023", "10") == (FISH, "source_code_crosswalk")
    assert direct_foodex2_label("afcd", "18101") == (MEAT, "source_code_crosswalk")
    assert direct_foodex2_label("cofid", "DG") == (VEGETABLES, "source_code_crosswalk")
    assert direct_foodex2_label("phyfoodcomp_1_0", "03") == (LEGUMES, "source_code_crosswalk")
    assert direct_foodex2_label("lesotho_fct_2006", "01") == (GRAINS, "source_code_crosswalk")


def test_ambiguous_native_groups_remain_unlabelled():
    assert direct_foodex2_label("ciqual", "meat, egg and fish") == ("", "unlabelled")
    assert direct_foodex2_label("cnf", "Dairy and Egg Products") == ("", "unlabelled")
    assert direct_foodex2_label("foodb", "Animal foods") == ("", "unlabelled")
    assert direct_foodex2_label("any", "Beverages") == ("", "unlabelled")


def test_exact_name_transfer_requires_unanimous_direct_labels():
    profiles = pd.DataFrame({
        "source_key": ["foodb", "foodb", "foodb", "foodb", "foodb", "foodb"],
        "food_group": ["Fruits", None, "Fruits", "Vegetables", None, "Beverages"],
        "exact_name_group_id": ["same", "same", "conflict", "conflict", "conflict", "same"],
    })
    labels = assign_foodex2_labels(profiles)
    assert labels.loc[1, "label_provenance"] == "exact_name_unanimous_transfer"
    assert labels.loc[1, "foodex2_level1"] != ""
    assert labels.loc[4, "foodex2_level1"] == ""
    assert labels.loc[4, "label_provenance"] == "unlabelled"
    assert labels.loc[5, "foodex2_level1"] == ""
