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

def test_exactly_sufficient_supply_does_not_scale_at_all():
    # The boundary the proportional rule turns on: scaling must happen ONLY when supply is
    # short. With released + reserve stock exactly equal to the gains, the factor is 1 and every
    # feasible request survives untouched -- not 0.9999 from a floating-point min().
    r = np.array([0., 30, 0, -10, 0.]); s = np.array([0., 0, 0, 100, 20])
    d, a = m.constrain_to_reserve(r, s)
    assert a['expansion_factor'] == 1.0
    np.testing.assert_array_equal(d[:4], r[:4])
    assert d[4] == -20

def test_one_hectare_short_scales_and_the_shortfall_is_recorded():
    # And just past the boundary it does scale, so the two tests bracket the condition.
    r = np.array([0., 31, 0, -10, 0.]); s = np.array([0., 0, 0, 100, 20])
    d, a = m.constrain_to_reserve(r, s)
    assert a['expansion_factor'] < 1.0
    assert d[1] < r[1]
    assert a['unmet_signed_ha'][1] > 0

def test_the_reserve_keeps_its_incoming_demand_in_the_record_but_not_in_the_target():
    # OtherNat's own request is overwritten by the balancing residual. The original must still be
    # legible afterwards, or a reader cannot tell a policy residual from a request that was met.
    r = np.array([0., 10, 0, -10, 777.]); s = np.array([0., 0, 0, 100, 100])
    d, a = m.constrain_to_reserve(r, s)
    assert a['original_ha'][4] == 777.
    assert d[4] == 0.
    assert a['reserve_target_is_policy_residual'] is True

def test_the_reserve_cannot_be_overdrawn():
    # The residual is not allowed to demand more OtherNat than is eligible.
    try:
        m.constrain_to_reserve(np.array([0., 500, 0, 0, 0.]), np.array([0., 0, 0, 0, 10.]))
    except ValueError as error:
        assert 'overdraw' in str(error).lower() or 'reserve' in str(error).lower()
    else:
        # Scaling may absorb it instead; then the target must respect the stock.
        d, _ = m.constrain_to_reserve(np.array([0., 500, 0, 0, 0.]), np.array([0., 0, 0, 0, 10.]))
        assert d[4] >= -10.0 - 1e-9

# Two real coarse cells that the ORIGINAL direct request-to-allocator path got wrong, carried here
# as numbers rather than as a dependency on the private evidence directory. They are regression
# cases for TWO DISTINCT defects, which must not be conflated:
#
#   Defect 1, PRE-EXISTING, in the original Cython allocator: donors were not restricted to
#   contracting classes and donor budgets were not tracked, so a cell asking for 2375.603 ha of
#   forest loss realised 7140.673 ha. A fine-pixel realisation fault.
#
#   Defect 2, OURS, introduced by the October 5 handoff: coarse-cell requests were summed to a
#   region/AEZ budget and redistributed from target and eligible stock alone, so even feasible
#   80/20 patterns came out 50/50. A coarse-budget fault.
#
# Fixing defect 1 does not license defect 2. These tests pin the budget half: a feasible
# coarse-cell request must pass through untouched while a genuinely infeasible one is capped.

def test_real_net_zero_2030_cell_preserves_its_feasible_requests():
    # net_zero 2030, cell (272,211). Urban asks to contract 1.2288 ha with ZERO eligible stock,
    # so that one request is capped. Everything else is feasible and must survive unchanged.
    original = np.array([-1.2288497950960597, -0.07139999999998073, 1887.7689193643935,
                         -339.1129962289146, 14.447557503843846])
    stock = np.array([0.0, 3781.6977, 0.0, 160205.9106, 11873.3381])
    target, audit = m.constrain_to_reserve(original, stock)
    # The feasible requests are untouched -- not rescaled, not redistributed, not averaged.
    assert target[1] == original[1]
    assert target[2] == original[2]
    assert target[3] == original[3]
    # Supply was sufficient, so nothing is scaled at all.
    assert audit['expansion_factor'] == 1.0
    # The single infeasible request is capped at its stock of zero, and the shortfall is recorded.
    assert target[0] == 0.0
    assert abs(audit['contraction_shortfall_ha'][0] - 1.2288497950960597) < 1e-9
    # OtherNat carries the balance; its own incoming request is replaced, not met.
    assert abs(target[4] - -1548.5845231354788) < 1e-6
    assert abs(sum(target)) < 1e-6

def test_real_current_policies_2040_cell_scales_because_supply_is_short():
    # current_policies 2040, cell (225,266) -- the real request, not a constructed one. Cropland
    # asks for 41371.67 ha against released land plus an OtherNat stock of 11522.38, so supply IS
    # short and the positive demand scales. The other regime, and why both cells are kept.
    original = np.array([-37.94610305750945, 41371.66811160474, -2118.0669734019184,
                         -5133.774775158884, 28.048258959964038])
    stock = np.array([9507.824931230107, 0.0, 15773.44572000188, 30472.9321816233,
                      11522.376304549576])
    target, audit = m.constrain_to_reserve(original, stock)
    assert abs(audit['expansion_factor'] - 0.4547112798405893) < 1e-12
    assert abs(target[1] - 18812.164156167888) < 1e-6
    # Contractions are NOT scaled; only positive demands are. All three pass through untouched.
    assert target[0] == original[0]
    assert target[2] == original[2]
    assert target[3] == original[3]
    # Cropland is the only positive demand here, so proportionality shows as the factor itself.
    assert abs(target[1] / original[1] - audit['expansion_factor']) < 1e-12
    # Nothing was capped: the shortfall is entirely a scaling effect, not an eligibility cap.
    assert all(v == 0.0 for v in audit['contraction_shortfall_ha'])
    assert abs(audit['unmet_signed_ha'][1] - 22559.503955436852) < 1e-6
    # OtherNat takes the balance, exactly to its eligible stock.
    assert abs(target[4] - -11522.376304549576) < 1e-6
    assert abs(sum(target)) < 1e-6
