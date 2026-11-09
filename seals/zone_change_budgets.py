"""Per-cell change budgets in hectares.

constrain_to_reserve serves the signed pixel allocation and never redistributes
across cells. strict_othernat_targets and audit_strict_allocation serve the
legacy calibrate path when strict_othernat_accounting is set.
"""
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


def _cell_windows(shape, fine_shape):
    rows, cols = shape
    height, width = fine_shape
    if height % rows or width % cols:
        raise ValueError('Fine and coarse grids must have an integer size ratio')
    dy, dx = height // rows, width // cols
    for row in range(rows):
        for col in range(cols):
            yield row, col, (slice(row * dy, (row + 1) * dy),
                             slice(col * dx, (col + 1) * dx))


def strict_othernat_targets(requests, before, hectares, mask, class_codes, reserve_index):
    """Balance OtherNat and reject any contraction exceeding unmasked starting stock.

    Stock availability is a necessary check. Suitability and pixel allocation
    still require an audit of the resulting map. Inputs are never modified.
    """
    requests = np.asarray(requests, dtype=float)
    codes = np.asarray(class_codes)
    if (requests.ndim != 3 or requests.shape[0] != len(codes)
            or len(set(codes)) != len(codes) or not 0 <= reserve_index < len(codes)):
        raise ValueError('Demand classes and reserve index must match unique class codes')
    if before.shape != hectares.shape or before.shape != mask.shape:
        raise ValueError('Starting map, hectares and mask must share a grid')
    if not np.isfinite(requests).all() or not np.isfinite(hectares).all() or (hectares <= 0).any():
        raise ValueError('Demands must be finite and pixel hectares positive and finite')
    targets = requests.copy()
    targets[reserve_index] = 0.
    targets[reserve_index] = -targets.sum(axis=0)
    records = []
    for row, col, window in _cell_windows(requests.shape[1:], before.shape):
        b, h, valid = before[window], hectares[window], mask[window] != 0
        stock = np.array([h[(b == code) & valid].sum() for code in codes])
        target = targets[:, row, col]
        shortfall = np.maximum(-target - stock, 0.)
        if (shortfall > 1e-6).any():
            raise ValueError(f'Insufficient land at coarse cell ({row}, {col}): '
                             f'class codes {codes[shortfall > 1e-6].tolist()}, '
                             f'shortfall ha {shortfall[shortfall > 1e-6].tolist()}. '
                             'GTAP demands were not reduced.')
        for i, code in enumerate(codes):
            records.append(dict(coarse_row=row, coarse_col=col, class_code=int(code),
                                original_ha=requests[i, row, col], target_ha=target[i],
                                available_ha=stock[i]))
    return targets, records


def audit_strict_allocation(targets, before, after, hectares, mask, class_codes):
    """Reject unmet targets, donor overdraw and changes to excluded pixels.

    Each coarse-cell class may differ by at most its largest unmasked pixel's
    area plus 1e-6 ha. That rounding tolerance does not relax stock checks.
    """
    codes = np.asarray(class_codes)
    if after.shape != before.shape:
        raise ValueError('Allocated map must share the starting grid')
    fixed = (mask == 0) | ~np.isin(before, codes)
    if not np.array_equal(before[fixed], after[fixed]):
        raise ValueError('Allocation changed masked or excluded land')
    if not np.isin(after[~fixed], codes).all():
        raise ValueError('Allocation introduced an unknown land class')
    records = []
    for row, col, window in _cell_windows(targets.shape[1:], before.shape):
        b, a, h = before[window], after[window], hectares[window]
        valid = mask[window] != 0
        tolerance = float(h[valid].max()) if valid.any() else 0.
        target = targets[:, row, col]
        actual = np.array([h[a == code].sum() - h[b == code].sum() for code in codes])
        losses = np.array([h[(b == code) & (a != code)].sum() for code in codes])
        overdraw = (losses > 0) & ((target >= 0) | (losses > -target + tolerance + 1e-6))
        if (np.abs(actual - target) > tolerance + 1e-6).any() or overdraw.any() or abs(actual.sum()) > 1e-6:
            raise ValueError(f'Allocation failed at coarse cell ({row}, {col}): '
                             f'residual ha {(actual - target).tolist()}, '
                             f'donor overdraw {codes[overdraw].tolist()}')
        for i, code in enumerate(codes):
            records.append(dict(coarse_row=row, coarse_col=col, class_code=int(code),
                                target_ha=target[i], realised_ha=actual[i],
                                residual_ha=actual[i] - target[i], tolerance_ha=tolerance))
    return records
