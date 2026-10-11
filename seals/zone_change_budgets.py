"""Cell-local reserve balancing for signed pixel allocation; never redistribute across cells."""
import numpy as np


def constrain_to_reserve(change, eligible_stock, reserve_index=4):
    """Cap contractions and share scarce supply across ALL positive demands.

    Stock and change are hectares on identical support. The reserve's incoming
    demand is retained in the audit but replaced with the balancing residual.
    Eligibility is supplied by the spatial caller, not inferred from raw area.
    This explicitly changes unmet targets; it is not feedback to an economic run.
    """
    original = np.asarray(change, dtype=np.float64)
    stock = np.asarray(eligible_stock, dtype=np.float64)
    if original.ndim != 1 or stock.shape != original.shape:
        raise ValueError('Demand and stock must be matching class vectors')
    if not np.isfinite(original).all() or not np.isfinite(stock).all() or (stock < 0).any():
        raise ValueError('Demand/stock must be finite and stocks nonnegative')
    if not 0 <= reserve_index < len(original):
        raise ValueError('Invalid reserve class')
    target = original.copy()
    target[reserve_index] = 0.
    contracting = target < 0
    target[contracting] = -np.minimum(-target[contracting], stock[contracting])
    contraction_limited = target.copy()
    expanding = target > 0
    gains = target[expanding].sum()
    released = -target[target < 0].sum()
    available = released + stock[reserve_index]
    factor = min(1., available / gains) if gains else 1.
    target[expanding] *= factor
    target[reserve_index] = -target.sum()
    eps = 32 * np.finfo(float).eps * max(1., np.abs(target).sum())
    if target[reserve_index] < -stock[reserve_index] - eps:
        raise ValueError('Reserve overdraw')
    return target, {
        'original_ha': original.tolist(), 'target_ha': target.tolist(),
        'unmet_signed_ha': (original - target).tolist(),
        'contraction_shortfall_ha': np.where(contracting, contraction_limited-original, 0.).tolist(),
        'eligible_stock_ha': stock.tolist(), 'expansion_factor': float(factor),
        'reserve_index': reserve_index,
        'reserve_target_is_policy_residual': True,
    }

