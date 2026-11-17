"""Every library task-tree builder must construct: each task function it names has to exist.

A builder is only a list of add_task calls, so the one way it rots is by naming a task that
was since renamed or removed, which surfaces as an AttributeError the first time someone
runs it. Building the tree on a bare ProjectFlow catches that without any data.
"""
import pytest

import hazelbean as hb

from seals import seals_initialize_project

BUILDER_NAMES = [
    "build_standard_task_tree",
    "build_complete_run_task_tree",
    "build_ken_task_tree",
    "build_standard_with_postprocessing_task_tree",
    "build_custom_coarse_algorithm_task_tree",
]


@pytest.mark.parametrize("builder_name", BUILDER_NAMES)
def test_builder_constructs(builder_name, tmp_path):
    p = hb.ProjectFlow(project_dir=str(tmp_path))
    p.run_in_parallel = 1  # Read at construction time by the trees' parallel iterator tasks.

    getattr(seals_initialize_project, builder_name)(p)

    assert p.allocation_task is not None
