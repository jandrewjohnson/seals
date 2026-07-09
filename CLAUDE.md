# seals_dev

SEALS — spatial land-use/land-cover change allocation model. Part of the
earth-economy devstack; builds on the shared **hazelbean** base library
(`import hazelbean as hb`).

Shared devstack guidance (reuse rule + ownership map + EE conventions pointer) is
imported below — before adding any general utility (paths, raster/vector ops,
ProjectFlow tasks), search hazelbean first and reuse/extend rather than duplicate.

@../../earth_economy_devstack/devstack_guidance.md

## Environment

Conda-based. Activate a conda environment with the stack's dependencies before
running code (each contributor may use a different env name — see your own
user-level config for yours).

## Package

Installable package name: `sealsmodel` (import as `seals`). Tests live in `seals_tests/`.
