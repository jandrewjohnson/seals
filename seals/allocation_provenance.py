"""Acceptance checks for stitching strict cell-residual allocation outputs."""
from pathlib import Path
import json
import numpy as np
import hazelbean as hb


def validated_tiles(target_dir, filename):
    """Require every planned tile and its accepted map/mask hashes before stitching."""
    root = Path(target_dir) / 'allocation_zones'
    blocks_path = root / 'global_processing_blocks_list.csv'
    if not blocks_path.is_file():
        raise RuntimeError(f'Missing planned allocation footprint: {blocks_path}')
    blocks = np.loadtxt(blocks_path, delimiter=',', dtype=int, ndmin=2)
    result = {}
    for block in blocks:
        tile = root / f'{block[0]}_{block[1]}'
        audit_path = tile / 'accepted.json'
        if not audit_path.is_file():
            raise RuntimeError(f'Missing strict tile acceptance: {audit_path}')
        audit = json.loads(audit_path.read_text())
        if audit.get('status') != 'accepted' or audit.get('mode') != 'cell_residual_strict':
            raise RuntimeError(f'Unaccepted allocation tile: {tile}')
        for name in (filename, 'cumulative_change_happened.tif'):
            path = tile / name
            if not path.is_file() or hb.hash_file_path(str(path)) != audit.get('outputs', {}).get(name):
                raise RuntimeError(f'Allocation output differs from acceptance audit: {path}')
        result[str(tile / filename)] = hb.hash_file_path(str(audit_path))
    if not result:
        raise RuntimeError('Strict stitching requires an explicit nonempty accepted footprint')
    return result


def check_stitched_resume(path, tiles, baseline):
    """Refuse an existing mosaic whose accepted sources or bytes have changed."""
    path = Path(path)
    audit_path = path.with_suffix('.allocation.json')
    if not path.exists():
        return
    if not audit_path.is_file():
        raise RuntimeError(f'Existing mosaic lacks strict provenance: {path}')
    audit = json.loads(audit_path.read_text())
    if audit.get('tiles') != tiles or audit.get('baseline') != baseline or audit.get('sha256') != hb.hash_file_path(str(path)):
        raise RuntimeError(f'Stitched map provenance differs: {path}')


def record_stitched(path, tiles, baseline):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f'Stitch did not produce a map: {path}')
    audit_path = path.with_suffix('.allocation.json')
    record = dict(tiles=tiles, baseline=baseline, sha256=hb.hash_file_path(str(path)))
    # This is called for every stitched map on every run, including maps the resume reused. Writing
    # an identical record replaces the file -- atomically, so with a new inode -- for no change,
    # which is work repeated on resume and makes an unchanged-resume check impossible to satisfy.
    # check_stitched_resume has already refused a record that disagrees, so one that equals this is
    # current and must be left alone.
    if audit_path.is_file():
        try:
            if json.loads(audit_path.read_text()) == record:
                return
        except (ValueError, OSError):
            pass
    temporary = audit_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(audit_path)
