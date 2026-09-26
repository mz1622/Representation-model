import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from foodcomp.research_r0 import weighted_median, scale_conflict, canonical_cells, typical_scales, ResearchData, score_predictions
from foodcomp.schema import normalize_unit

class ResearchProtocolTests(unittest.TestCase):
    def profiles(self):
        return pd.DataFrame({"profile_id":["p0","p1","p2"],"profile_index":[0,1,2],
            "partition":["train","train","validation"],"source_key":["s0","s1","s0"],"exact_name_group_id":["g0","g0","g1"]})

    def test_raw_median_is_evaluator_truth_not_inverse_log_median(self):
        tokens=pd.DataFrame({"profile_id":["p0","p0"],"axis_index":[0,0],
            "measurement_id":["m0","m1"],"normalized_value_g_per_100g":[.1,100.]})
        cells=canonical_cells(tokens,self.profiles())
        self.assertAlmostEqual(cells.value.iloc[0],50.05)
        self.assertGreater(cells.aggregation_gap.iloc[0],40)
        self.assertTrue(cells.quarantined.iloc[0])

    def test_zero_is_observed_not_missing_or_scale_conflict(self):
        self.assertFalse(scale_conflict([0],[100])[0])
        self.assertTrue(scale_conflict([.0001],[100])[0])
        self.assertFalse(scale_conflict([1],[100])[0])

    def test_both_unit_spellings_are_correct_in_current_code(self):
        self.assertEqual(normalize_unit("mg/100g","mg/100g")["conversion_factor"],.001)
        self.assertEqual(normalize_unit("mg/100 g","mg/100 g")["conversion_factor"],.001)

    def test_weighted_median_and_invalid_weights(self):
        self.assertEqual(weighted_median([1,10,100],[1,3,1]),10)
        with self.assertRaises(ValueError): weighted_median([1],[0])

    def test_scale_fitting_does_not_use_validation_values(self):
        cells=pd.DataFrame({"profile_id":["p0","p1","p2"],"axis_index":[0,0,0],"value":[2,4,1e9],
             "partition":["train","train","validation"],"source_key":["s0","s1","s0"],"exact_name_group_id":["g0","g0","g1"]})
        scale,_=typical_scales(cells,1)
        self.assertEqual(scale[0],2)
        cells.loc[2,"value"]=1e-12
        self.assertEqual(typical_scales(cells,1)[0][0],2)

    def fake_data(self):
        d=object.__new__(ResearchData)
        d.profiles=self.profiles()
        d.axes=pd.DataFrame({"axis_index":[0,1,2],"canonical_name":["a","b","c"],
            "mask_family":["f","f","g"],"loss_group":["nutrition"]*3})
        d.raw=np.array([[0,5,np.nan],[3,4,7],[1,2,0]],float)
        d.observed=np.isfinite(d.raw);d.scale=np.ones(3);d.values=np.nan_to_num(np.log1p(d.raw))
        d.families=np.array(["f","f","g"]);d.targets=np.array([0,1,2])
        d.jobs=pd.DataFrame({"profile_index":[2]*3,"axis_index":[0,1,2],"target":[1.,2.,0.]})
        return d

    def test_hidden_labels_cannot_change_context_and_family_is_hidden(self):
        d=self.fake_data(); v,m=d.context([2],0)
        self.assertFalse(m[0,0]);self.assertFalse(m[0,1]);self.assertTrue(m[0,2])
        d.values[2,:2]=999
        v2,m2=d.context([2],0)
        np.testing.assert_array_equal(v,v2);np.testing.assert_array_equal(m,m2)

    def test_name_only_has_no_numeric_values_or_observedness(self):
        v,m=self.fake_data().context([0,1,2],mode="name_only")
        self.assertFalse(m.any());self.assertEqual(v.sum(),0)

    def test_perfect_predictions_and_zero_label(self):
        d=self.fake_data()
        p=d.jobs.drop(columns="target").assign(prediction=[1.,2.,0.])
        metrics,_,_=score_predictions(d,p)
        self.assertEqual(metrics["nutrition"]["scaled_log_mae"],0)

    def test_missing_extra_duplicate_nonfinite_or_relabelled_jobs_fail(self):
        d=self.fake_data()
        p=d.jobs.drop(columns="target").assign(prediction=[1.,2.,0.])
        variants=[p.iloc[:2],pd.concat([p,p.iloc[:1]]),p.assign(prediction=np.nan),p.assign(target=123)]
        for variant in variants:
            with self.subTest(variant=str(variant)):
                with self.assertRaises(ValueError): score_predictions(d,variant)

    def test_fixed_visibility_is_shared_deterministic_and_label_independent(self):
        d=self.fake_data()
        x,m=d.context([0,1,2],2,visible_fraction=.5)
        d.values[:]=123
        _,m2=d.context([0,1,2],2,visible_fraction=.5)
        np.testing.assert_array_equal(m,m2)

if __name__=="__main__":
    unittest.main()
