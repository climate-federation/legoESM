"""CLUBB cloud-fraction -> radiation PIPELINE-path contract (build_physics_pipeline
/ compute_radiation_core / model_driver rollout).

The main AMIP path runs physics through ``PhysicsPipeline`` (not the combined.py
accumulator): diagnostic CLUBB writes its PDF cloud fraction onto
``PhysicsOutput.cloud_fraction`` in ``physics_step_no_rad``; the NEXT
``compute_radiation_core`` reads it (a per-step carry threaded through
``_run_per_step``) and routes it to the cloud optics.  These pin the low-cost
structural contract (field present + signatures + import health) so a refactor
that drops the carry hand-off is caught without a full model integration run.
"""

from __future__ import annotations

import inspect

from legoesm.core.physics_output import PhysicsOutput


def test_physics_output_has_cloud_fraction_field_default_none():
    """The carry slot exists and defaults None (byte-identical for closures with
    no PDF cloud => radiation keeps the RH grid-scale path).  PhysicsOutput's
    leading tendency fields are required positionals, so check the field/default
    via the NamedTuple metadata rather than a bare construction."""
    assert "cloud_fraction" in PhysicsOutput._fields
    assert PhysicsOutput._field_defaults.get("cloud_fraction", "MISSING") is None
    # cloud_fraction is the LAST field (appended), so a positional build that
    # omits it still works — the default None applies.
    assert PhysicsOutput._fields[-1] == "cloud_fraction"


def test_compute_radiation_core_accepts_cloud_fraction():
    """compute_radiation_core takes the per-step cloud-fraction carry (routed to
    compute_cloud_properties' override)."""
    from legoesm.driver.physics_pipeline import PhysicsPipeline
    sig = inspect.signature(PhysicsPipeline.compute_radiation_core)
    assert "cloud_fraction" in sig.parameters
    assert sig.parameters["cloud_fraction"].default is None


def test_step_unified_builder_and_driver_import_clean():
    """The pipeline + model_driver modules import with the carry threading in
    place (guards against a mis-threaded tuple / signature breaking import)."""
    import legoesm.driver.physics_pipeline as pp
    import legoesm.driver.model_driver as md
    assert hasattr(pp, "PhysicsPipeline")
    assert hasattr(md, "ModelDriver")
