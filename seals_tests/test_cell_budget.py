import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('budget',Path(__file__).parents[1]/'seals/zone_change_budgets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_othernat_is_residual_and_feasible_classes_unchanged():
    r=np.array([0.,20,30,-10,999.]);s=np.array([0.,0,0,100,100])
    d,a=m.constrain_to_reserve(r,s)
    np.testing.assert_array_equal(d[:4],r[:4]);assert d[4]==-40;assert a['expansion_factor']==1

def test_scarcity_scales_all_positive_classes_proportionally():
    r=np.array([10.,20,30,-10,999.]);s=np.array([0.,0,0,100,20])
    d,a=m.constrain_to_reserve(r,s)
    np.testing.assert_array_equal(d,[5,10,15,-10,-20]);assert a['expansion_factor']==.5

def test_contraction_cannot_exceed_eligible_stock():
    d,a=m.constrain_to_reserve(np.array([0,0,0,-100,0.]),np.array([0,0,0,20,0.]))
    np.testing.assert_array_equal(d,[0,0,0,-20,20]);assert a['contraction_shortfall_ha'][3]==80
