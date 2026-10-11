import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('signed',Path(__file__).parents[1]/'seals/signed_allocation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_expanding_classes_and_zero_target_never_donate():
    b=np.array([[1,2,3,4,5,5,5,5,5,5]]);h=np.ones_like(b,dtype=float);mask=np.ones_like(b);r=np.tile(np.arange(b.size),(5,1)).reshape(5,*b.shape).astype(float)
    x,a=m.allocate_cell(b,h,mask,r,np.array([0,2,2,-1,-3.]))
    assert a['accepted'];np.testing.assert_array_equal(x[0,:3],b[0,:3]);assert a['loss'][:3]==[0,0,0]

def test_insufficient_pixels_fail_closed():
    b=np.array([[5,6]]);h=np.ones_like(b,dtype=float);mask=np.ones_like(b);r=np.zeros((5,1,2))
    with pytest.raises(RuntimeError):m.allocate_cell(b,h,mask,r,np.array([0,5,0,0,-5.]))

def test_mask_and_excluded_land_preserved():
    b=np.array([[5,5,6,7,5,5]]);mask=np.array([[0,1,1,1,1,1]]);h=np.ones_like(b,dtype=float);r=np.zeros((5,1,6))
    x,a=m.allocate_cell(b,h,mask,r,np.array([0,2,0,0,-2.]))
    assert a['accepted'];np.testing.assert_array_equal(x[0,[0,2,3]],b[0,[0,2,3]])

def test_unbalanced_rejected():
    with pytest.raises(ValueError):m.allocate_cell(np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),np.ones((5,1,1)),np.ones(5))

def test_multiple_small_expansions_do_not_starve_larger_recipient():
    b=np.full((1,100),5);h=np.full((1,100),10.);mask=np.ones_like(b);r=np.tile(np.arange(100),(5,1)).reshape(5,1,100).astype(float)
    d=np.array([1.,101.,797.,101.,-1000.])
    x,a=m.allocate_cell(b,h,mask,r,d)
    assert a['accepted'];assert max(abs(np.array(a['residual'])))<=10+1e-6

def test_grid_preserves_distinct_feasible_cell_patterns_and_balances_reserve():
    import sys, types
    # Load the real sibling budget module without importing the raster stack.
    root=Path(__file__).parents[1]/'seals'
    pkg=types.ModuleType('isolated_seals');pkg.__path__=[str(root)];sys.modules['isolated_seals']=pkg
    spec=importlib.util.spec_from_file_location('isolated_seals.signed_allocation',root/'signed_allocation.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    b=np.full((10,20),5);h=np.ones_like(b,dtype=float);mask=np.ones_like(b);layers=np.ones((1,10,20));coef=np.ones((5,1));types_=np.array([2]);r=np.zeros((5,1,2));r[1,0]=[80,20]
    x,_,_,audit=mod.allocate_grid(r,b,h,mask,layers,coef,types_)
    assert (x[:,:10]==2).sum()==80;assert (x[:,10:]==2).sum()==20
    assert [v['budget']['target_ha'][4] for v in audit]==[-80,-20]


def test_multiple_donor_rounding_preserves_feasible_contractions():
    # Fixed counterexample: an exact whole-pixel witness exists. The original
    # greedy candidate left grass 1.925 ha short with a 1.050 ha tolerance.
    rng=np.random.default_rng(20261012)
    for _ in range(6):
        b=rng.choice([1,2,3,4],size=(8,8));h=rng.uniform(.95,1.05,b.shape);mask=np.ones_like(b)
        witness=b.copy();witness[rng.random(b.shape)<rng.uniform(.1,.9)]=5
        demand=np.array([h[witness==k].sum()-h[b==k].sum() for k in range(1,6)])
        ranks=rng.random((5,*b.shape))
    assert m.validate_cell(b,witness,h,mask,demand)['accepted']
    after,audit=m.allocate_cell(b,h,mask,ranks,demand)
    assert audit['accepted']
    assert np.all((after==b)|(after==5))
