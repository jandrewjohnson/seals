"""Own interpolated coarse states in a separate ProjectFlow output directory."""
from pathlib import Path
import json
import os
import shutil
import hazelbean as hb
from seals import seals_utils


def coarse_interpolated_states(p):
    """Publish the fine-base-year state, including the common reference state.

    Only this task writes interpolation products. The proportion source tree is
    read-only; downstream readers resolve this output first for generated years.
    """
    if not p.run_this:return
    root=Path(p.cur_dir);root.mkdir(parents=True,exist_ok=True)
    for _,row in p.scenarios_df.iterrows():
        seals_utils.assign_df_row_to_object_attributes(p,row)
        if p.scenario_type=='baseline':continue
        raw=row['key_base_year'];year=int(raw[0] if isinstance(raw,(tuple,list)) else str(raw).split()[0])
        parts=(p.exogenous_label,p.climate_label,p.model_label,p.counterfactual_label)
        source=Path(p.coarse_simplified_proportion_dir).joinpath(*parts)
        brackets=seals_utils.bracketing_coarse_years(str(source),year)
        if brackets is None and year not in seals_utils.coarse_years_available(str(source)):
            raise ValueError(f'Cannot bracket coarse base year {year}: {source}')
        correspondence=hb.utils.get_reclassification_dict_from_df(p.get_path(p.coarse_correspondence_path,possible_dirs=[p.input_dir,p.base_data_dir]),'src_id','dst_id','src_label','dst_label')
        for label in correspondence['dst_labels']:
            template=f'{label}_prop_{p.exogenous_label}_{p.climate_label}_{p.model_label}_{p.counterfactual_label}_{{year}}.tif'
            destinations=[(root.joinpath(*parts,str(year)),template)]
            if p.scenario_type=='baseline_ignore_dependencies':
                destinations.append((root/p.exogenous_label/p.model_label/str(year),f'{label}_prop_{p.exogenous_label}_{p.model_label}_{{year}}.tif'))
            inputs=[source/str(y)/template.format(year=y) for y in (brackets or [year])]
            hashes={str(q):hb.hash_file_path(str(q)) for q in inputs}
            for directory,name in destinations:
                out=directory/name.format(year=year);manifest=out.with_suffix('.json')
                signature=dict(inputs=hashes,target_year=year,brackets=brackets,implementation=[hb.hash_file_path(__file__),hb.hash_file_path(seals_utils.__file__)])
                superseded=[];marker=out.with_name(out.name+'.publishing.json')
                # Recovery for an interruption between the two replacements below. The marker
                # exists only inside that window, so finding one means the previous process died
                # mid-publication and the raster and its manifest may disagree. That is a known
                # state, not corruption: recompute, which is idempotent, rather than raising on
                # the hash mismatch it would otherwise look like.
                already_superseded=False
                if marker.exists():
                    interrupted=json.loads(marker.read_text())
                    print('coarse_interpolated_states: recovering interrupted publication of %s '
                          '(prior version at %s)'%(out,interrupted.get('superseded_path')),flush=True)
                    kept=Path(interrupted['superseded_path']) if interrupted.get('superseded_path') else None
                    if kept is not None and kept.exists():
                        # The slot ALREADY holds the version this attempt superseded, so the
                        # rebuild below must not move anything into it again. Without this the
                        # kill-between-raster-and-manifest case destroys the saved original: `out`
                        # holds the NEW raster by then, and moving that aside overwrites the
                        # original the slot was keeping.
                        already_superseded=True
                        if not out.exists():
                            # Died between moving the prior version aside and publishing the new
                            # one. COPY it back rather than moving it, so the preserved version
                            # stays in its slot even if this recovery is itself interrupted.
                            shutil.copy2(kept,out)
                elif out.exists():
                    if manifest.exists():
                        previous=json.loads(manifest.read_text())
                        current=json.loads(json.dumps(signature))
                        if previous['signature']==current and previous['output_sha256']==hb.hash_file_path(str(out)):
                            continue
                        # A mismatch has two possible causes and they deserve opposite answers.
                        # If the INPUTS, the bracketing years or the output's own bytes changed,
                        # something rewrote data this task already consumed or produced: raise,
                        # because silently recomputing would hide an upstream fault. If only the
                        # IMPLEMENTATION hashes differ, someone edited this module or seals_utils,
                        # which is ordinary and must not brick the project directory -- the
                        # signature carries both file hashes, so any edit invalidated every
                        # manifest and the previous behaviour then aborted every tile.
                        if (previous['signature'].get('inputs')!=current['inputs']
                                or previous['signature'].get('brackets')!=current['brackets']
                                or previous['signature'].get('target_year')!=current['target_year']
                                or previous['output_sha256']!=hb.hash_file_path(str(out))):
                            raise RuntimeError(f'Interpolation provenance differs: {out}')
                        # Carry the record of what was here into the new manifest. The raster
                        # itself is recomputable from the signature's inputs, so keeping a
                        # timestamped copy of every superseded version would multiply files
                        # across ~300,000 tile directories against a file-count quota already
                        # at 92%; keeping the provenance costs nothing and is what makes the
                        # replacement auditable.
                        superseded=previous.get('superseded',[])+[dict(
                            signature=previous['signature'],output_sha256=previous['output_sha256'])]
                    else:
                        superseded=[dict(signature=None,output_sha256=hb.hash_file_path(str(out)),
                                         note='output existed with no manifest')]
                    # A derived raster whose provenance no longer matches is a STALE CACHE, so
                    # recompute it. Raising here, as this did, made every existing project
                    # directory permanently unusable after any edit to this module or to
                    # seals_utils -- the signature carries their file hashes -- and in a parallel
                    # run the first tile aborted everything. It also never self-healed: the same
                    # test failed on every rerun with unchanged code once a stale output existed.
                    # Contrast allocation_audit, where a provenance mismatch MUST raise, because
                    # an acceptance record is evidence rather than a cache and overwriting one
                    # would destroy the proof that a map was physically checked.
                    print(f'coarse_interpolated_states: recomputing stale {out}',flush=True)
                directory.mkdir(parents=True,exist_ok=True)
                # Build beside the target and replace atomically, so nothing is unlinked before
                # its replacement exists: an interrupted recompute leaves the previous raster and
                # its manifest intact rather than a hole where an output used to be.
                # The suffix goes BEFORE the extension: GDAL picks its driver from the
                # extension, so a '.tif.rebuilding' name silently writes nothing at all.
                staging=out.with_name(out.stem+'.rebuilding'+out.suffix)
                if brackets:
                    values,match=seals_utils.interpolate_coarse_state_at_year(str(source),year,brackets,template)
                    hb.save_array_as_geotiff(values,str(staging),match)
                else:shutil.copy2(inputs[0],staging)
                digest=hb.hash_file_path(str(staging))
                # Keep the PRIOR RASTER, not merely its hash. A hash cannot reconstruct a file,
                # and the version being replaced was produced by an implementation this checkout
                # no longer contains, so it is not recomputable from the inputs either. One slot
                # per output, overwritten each time, bounds the cost: this task writes per
                # scenario/year/class -- roughly 1,700 outputs for the full run -- not per tile,
                # so the slot adds ~1,700 files, unlike the ~300,000-tile allocation tree.
                kept_path=None if already_superseded else (
                    out.with_name(out.stem+'.superseded'+out.suffix) if out.exists() else None)
                slot=out.with_name(out.stem+'.superseded'+out.suffix)
                marker.write_text(json.dumps(dict(output=str(out),manifest=str(manifest),
                    superseded_path=str(slot) if (kept_path or (already_superseded and slot.exists())) else None,
                    new_output_sha256=digest,signature=signature),indent=2))
                if kept_path is not None:
                    os.replace(out,kept_path)
                    superseded=(superseded or [])
                    superseded.append(dict(path=str(kept_path),output_sha256=hb.hash_file_path(str(kept_path))))
                os.replace(staging,out)
                record=dict(signature=signature,output_sha256=digest)
                if superseded:record['superseded']=superseded
                manifest_staging=manifest.with_suffix('.json.tmp')
                manifest_staging.write_text(json.dumps(record,indent=2))
                os.replace(manifest_staging,manifest)
                marker.unlink()
