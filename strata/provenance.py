"""Lightweight provenance constants shared by data gates.

This module intentionally has no GIS imports. Training-time validation runs in
the model environment, so importing a list of reserved evaluation identities
must not initialize GeoBenchX or geopandas.
"""

RESERVED_TASK_IDS: frozenset[str] = frozenset({
    "TASK_250309_135125_702855",
    "TASK_250309_135125_476358",
    "TASK_250309_135125_530802",
    "TASK_250309_135125_526765",
    "TASK_250309_135125_901809",
    "TASK_250309_135125_596430",
    "TASK_250309_135125_476053",
    "TASK_250309_135125_918547",
    "TASK_250309_135125_932053",
})
