"""Recompute a test run from scratch and compare its outputs with the reference recorded on this machine.

    python scripts/check_baseline.py allocation     the pared Rwanda run of run_seals.py
    python scripts/check_baseline.py calibration    the pared Rwanda run of run_seals_calibration.py

The first run of each writes the reference, output_checksums_reference.json in the project dir. Every
later run recomputes everything and lists each raster or coefficient table that differs from it. To
record a new reference, after a change that is meant to alter results, delete that file and run again.
"""
import hashlib
import json
import os
import sys

import hazelbean as hb
import pandas as pd

# The guard is required: the runs start worker processes, which import this file again.
if __name__ == "__main__":
    pipeline = sys.argv[1]
    if pipeline == "allocation":
        import seals.run_seals as run_file
        scenario_definitions_filename = "standard_scenarios_test.csv"
        tasks_to_skip = None
    elif pipeline == "calibration":
        import seals.run_seals_calibration as run_file
        scenario_definitions_filename = "calibration_scenarios_test.csv"
        tasks_to_skip = run_file.projection_task_names
    else:
        raise ValueError("Expected allocation or calibration, got " + pipeline)

    project_dir = os.path.join(os.path.expanduser("~"), "Files", "seals", "projects", "seals_" + pipeline + "_baseline_test")
    p = hb.ProjectFlow(project_dir=project_dir, run_mode="fresh_intermediate")
    # The definitions CSVs are beside the run file, not beside this script.
    p.script_dir = os.path.dirname(os.path.abspath(run_file.__file__))
    p.input_template_dir = os.path.join(p.script_dir, "input_template")
    p.scenario_definitions_filename = scenario_definitions_filename
    p.tasks_to_skip = tasks_to_skip
    run_file.run_project(p)

    # One checksum per raster and per coefficient table. A table is hashed on its coefficient columns only, since its data_location column holds absolute paths.
    checksums = {}
    for current_dir, _, file_names in os.walk(p.intermediate_dir):
        for file_name in sorted(file_names):
            path = os.path.join(current_dir, file_name)
            if file_name.endswith(".tif"):
                values = hb.as_array(path)
            elif file_name.startswith("trained_coefficients") and file_name.endswith(".csv"):
                df = pd.read_csv(path)
                values = df[[column for column in df.columns if column.startswith("class_")]].to_numpy(dtype="float64")
            else:
                continue
            checksums[os.path.relpath(path, p.intermediate_dir).replace(os.sep, "/")] = hashlib.md5(values.tobytes()).hexdigest()

    reference_path = os.path.join(project_dir, "output_checksums_reference.json")
    if not hb.path_exists(reference_path):
        with open(reference_path, "w") as f:
            json.dump(checksums, f, indent=1, sort_keys=True)
        print ("Recorded a new reference of " + str(len(checksums)) + " outputs at " + reference_path)
    else:
        with open(reference_path) as f:
            reference = json.load(f)
        differing = sorted(key for key in checksums if key in reference and checksums[key] != reference[key])
        missing = sorted(key for key in reference if key not in checksums)
        added = sorted(key for key in checksums if key not in reference)
        for label, keys in [("DIFFERS", differing), ("MISSING", missing), ("NEW", added)]:
            for key in keys:
                print (label + " " + key)
        print (str(len(checksums)) + " outputs: " + str(len(differing)) + " differ, " + str(len(missing)) + " missing, " + str(len(added)) + " new, against " + reference_path)
        if differing or missing or added:
            sys.exit(1)
