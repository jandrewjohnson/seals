"""Isolated, fail-closed signed allocation candidate. No cross-cell redistribution."""
import numpy as np


def suitability_ranks(before, layers, coefficients, types, mask):
    ranks = np.zeros((len(coefficients), *before.shape), dtype=float)
    for c in range(len(coefficients)):
        for j, kind in enumerate(types):
            if kind == 2 and coefficients[c, j] != 0:
                ranks[c] += coefficients[c, j] * layers[j]
        for j, kind in enumerate(types):
            if kind == 1:
                ranks[c] *= 1 - (1 - coefficients[c, j]) * layers[j]
        ranks[c] *= mask
        ranks[c] *= before != c + 1
    ranks *= -1
    ranks[(ranks == 0) | ((ranks < 1e-14) & (ranks > -1e-14)) | (ranks < -9999999999999999)] = 9e9
    return ranks


def validate_cell(before, after, hectares, mask, demand):
    codes = np.arange(1, 6)
    tolerance = float(hectares[mask != 0].max()) if np.any(mask) else 0.
    delta = np.array([hectares[after == k].sum() - hectares[before == k].sum() for k in codes])
    loss = np.array([hectares[(before == k) & (after != k)].sum() for k in codes])
    overdraw = (loss > 0) & ((demand >= 0) | (loss > -demand + tolerance + 1e-6))
    failures = []
    if np.any(abs(delta - demand) > tolerance + 1e-6): failures.append('class_residual')
    if overdraw.any(): failures.append('donor_overdraw')
    if not np.array_equal(after[mask == 0], before[mask == 0]): failures.append('mask')
    excluded = ~np.isin(before, codes)
    if not np.array_equal(after[excluded], before[excluded]): failures.append('excluded_class')
    if abs(delta.sum()) > 1e-6: failures.append('conservation')
    return dict(accepted=not failures, failures=failures, delta=delta.tolist(), loss=loss.tolist(), residual=(delta-demand).tolist(), tolerance_ha=tolerance)


def allocate_cell(before, hectares, mask, ranks, demand):
    """Round-robin suitability ranking with explicit donor/receiver budgets.

    Uses the same conservative common suitability pool as the audited budget.
    Each original pixel changes at most once. A failed final audit raises; a
    greedy failure is not a mathematical infeasibility certificate.
    """
    before = np.asarray(before); demand = np.asarray(demand, dtype=float)
    if demand.shape != (5,) or ranks.shape != (5, *before.shape): raise ValueError('Five-class aligned arrays required')
    if not all(np.isfinite(x).all() for x in (demand, hectares, ranks)) or np.any(hectares <= 0): raise ValueError('Invalid inputs')
    if abs(demand.sum()) > 1e-6: raise ValueError('Unbalanced demand')
    recipients = np.where(demand > 1e-8)[0]
    common = (mask != 0) & np.all(ranks[recipients] < 999999999., axis=0)
    after = before.copy(); left = demand.copy(); changed = np.zeros(before.size, dtype=bool)
    flat = before.ravel(); area = hectares.ravel(); permitted = common.ravel()
    order = {c:np.argsort(ranks[c], axis=None) for c in recipients}
    pos = {c:0 for c in recipients}
    tol = float(hectares[mask != 0].max()) if np.any(mask) else 0
    for _ in range(before.size):
        progressed = False
        for c in recipients:
            if left[c] <= 0: continue
            while pos[c] < before.size:
                j = order[c][pos[c]]; pos[c] += 1
                donor = int(flat[j])-1
                if not permitted[j] or changed[j] or donor < 0 or donor >= 5 or demand[donor] >= 0: continue
                if left[donor] >= 0 or area[j] > -left[donor] + tol + 1e-6: continue
                # A recipient may overshoot by no more than one eligible pixel.
                if area[j] > left[c] + tol + 1e-6: continue
                after.ravel()[j] = c+1; changed[j] = True
                left[c] -= area[j]; left[donor] += area[j]
                progressed = True; break
        if not progressed: break
    # Whole-pixel overshoots of several recipients can starve the final one.
    # Reassign already converted donor pixels within this cell, strictly reducing
    # squared class residuals; original donor totals and eligibility do not change.
    for _ in range(before.size):
        actual = np.array([hectares[after == k].sum() - hectares[before == k].sum() for k in range(1, 6)])
        residual = actual - demand
        if np.all(abs(residual) <= tol + 1e-6): break
        moved = False
        for receiver in sorted(recipients, key=lambda c: residual[c]):
            if residual[receiver] >= -tol - 1e-6: continue
            for j in order[receiver]:
                if not changed[j] or not permitted[j]: continue
                owner = int(after.ravel()[j]) - 1
                if owner == receiver or residual[owner] <= 0: continue
                # Exact change in two-class squared residual must be negative.
                if area[j] >= residual[owner] - residual[receiver] - 1e-9: continue
                after.ravel()[j] = receiver + 1
                moved = True; break
            if moved: break
        if not moved: break
    # Several donors can each overshoot by a pixel, leaving another donor
    # under-contracted even when recipient totals are already met. Exchange an
    # already selected donor pixel for an unused pixel of the underfilled donor,
    # keeping the same recipient. No cross-cell or previous-interval moves.
    for _ in range(before.size):
        actual = np.array([hectares[after == k].sum() - hectares[before == k].sum() for k in range(1, 6)])
        residual = actual - demand
        under = [c for c in range(5) if demand[c] < 0 and residual[c] > tol + 1e-6]
        if not under: break
        moved = False
        for donor in sorted(under, key=lambda c: -residual[c]):
            for receiver in recipients:
                additions = [j for j in order[receiver] if permitted[j] and not changed[j] and flat[j] == donor+1]
                removals = [j for j in reversed(order[receiver]) if changed[j] and after.ravel()[j] == receiver+1
                            and demand[int(flat[j])-1] < 0 and residual[int(flat[j])-1] < 0]
                for j in additions:
                    for k in removals:
                        other = int(flat[k])-1
                        old = residual[[donor, other, receiver]]
                        new = old + np.array([-area[j], area[k], area[j]-area[k]])
                        if abs(new[2]) > tol + 1e-6: continue
                        if np.dot(new,new) >= np.dot(old,old)-1e-9: continue
                        after.ravel()[k] = flat[k]; changed[k] = False
                        after.ravel()[j] = receiver+1; changed[j] = True
                        moved = True; break
                    if moved: break
                if moved: break
            if moved: break
        if not moved: break
    audit = validate_cell(before, after, hectares, mask, demand)
    if not audit['accepted']: raise RuntimeError('Signed allocation failed: ' + str(audit))
    return after, audit


def cell_windows(requests, before):
    """Yield aligned coarse-cell windows without performing I/O."""
    requests = np.asarray(requests, dtype=float)
    if requests.ndim != 3 or requests.shape[0] != 5:
        raise ValueError('Expected five-class coarse demand cube')
    nr, nc = requests.shape[1:]; height, width = before.shape
    if height % nr or width % nc:
        raise ValueError('Fine/coarse grids not aligned')
    rh, cw = height // nr, width // nc
    if rh != cw:
        raise ValueError('Fine/coarse resolution ratio differs by axis')
    for cr in range(nr):
        for cc in range(nc):
            yield cr, cc, np.s_[cr*rh:(cr+1)*rh, cc*cw:(cc+1)*cw]


def budget_grid(requests, before, hectares, mask, layers, coefficients, types):
    """Compute cell-local targets and stocks; the ProjectFlow task owns persistence."""
    from .zone_change_budgets import constrain_to_reserve
    targets = np.zeros_like(requests, dtype=float); audits = []
    for cr, cc, window in cell_windows(requests, before):
        b=before[window]; h=hectares[window]; m=mask[window]
        ranks=suitability_ranks(b,layers[:,window[0],window[1]],coefficients,types,m)
        req=requests[:,cr,cc]
        receivers=np.where(req[:4]>1e-8)[0]
        common=(m!=0)&np.all(ranks[receivers]<999999999.,axis=0)
        stocks=np.array([h[(b==k)&common].sum() for k in range(1,6)])
        target,budget_audit=constrain_to_reserve(req,stocks)
        if target[4]>1e-8:
            common &= ranks[4]<999999999.
            stocks=np.array([h[(b==k)&common].sum() for k in range(1,6)])
            target,budget_audit=constrain_to_reserve(req,stocks)
        targets[:,cr,cc]=target
        audits.append(dict(coarse_row=cr,coarse_col=cc,budget=budget_audit))
    return targets, audits


class AllocationFailure(RuntimeError):
    """Carry exact failing inputs to an audit task; never write from the allocator."""
    def __init__(self, cell, inputs, cause):
        super().__init__(f'Coarse cell {cell} rejected: {cause}')
        self.cell = cell
        self.inputs = inputs


def allocate_budget_grid(targets, before, hectares, mask, layers, coefficients, types):
    """Allocate explicit persisted budgets, independently of budget preparation."""
    after=before.copy(); audits=[]
    for cr,cc,window in cell_windows(targets,before):
        b=before[window]; h=hectares[window]; m=mask[window]
        cell_layers=layers[:,window[0],window[1]]
        ranks=suitability_ranks(b,cell_layers,coefficients,types,m)
        try:
            result,pixel_audit=allocate_cell(b,h,m,ranks,targets[:,cr,cc])
        except Exception as error:
            raise AllocationFailure((cr,cc), dict(before=b,hectares=h,mask=m,
                layers=cell_layers,coefficients=coefficients,types=types,
                target=targets[:,cr,cc]),error) from error
        after[window]=result
        audits.append(dict(coarse_row=cr,coarse_col=cc,pixels=pixel_audit))
    changed=(after!=before).astype(np.int64)
    changes=np.array([(changed & (after==k)).astype(np.int64) for k in range(1,6)])
    return after,changes,changed,audits


def allocate_grid(requests, before, hectares, mask, layers, coefficients, types):
    """Pure convenience composition; production tasks persist budgets before allocation."""
    targets,budgets=budget_grid(requests,before,hectares,mask,layers,coefficients,types)
    after,changes,changed,pixels=allocate_budget_grid(targets,before,hectares,mask,layers,coefficients,types)
    audits=[dict(b, pixels=a['pixels']) for b,a in zip(budgets,pixels)]
    return after,changes,changed,audits


def input_signature(*arrays):
    """Bind strict tile reuse to exact allocation inputs and implementation."""
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    for name in ('signed_allocation.py', 'zone_change_budgets.py'):
        digest.update((Path(__file__).parent/name).read_bytes())
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.dtype.str, a.shape)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()
