"""Shared rollout-curriculum schedule for WB + AIMIP training loops.

One source of truth for the NeuralGCM-style short-lead-first rollout
curriculum used by BOTH the spectral dycore-mode training loop
(``neural_gcm_spectral._train_spectral_loop``) and the sfno_full macro-step
loop (``build_sfno_curriculum_epoch_plan`` delegates here), plus the
stage-level metadata (per-stage LR scale, pushforward prefix) consumed by the
unified campaign driver.

Design references (docs/superpowers/specs/2026-07-19-unified-wb-aimip-training-design.md):
- horizon ladder + per-stage LR restart: GraphCast (arXiv:2212.12794) fine-tune
  1->12 steps at tiny constant LR; GDPS fine-tune (arXiv:2408.14587) shows
  per-stage restarts at larger LR also work.
- pushforward (no-grad rollout prefix): Brandstetter et al. (arXiv:2202.03376).

This module is deliberately JAX-free (pure schedule math) so it imports fast
and is testable on any node.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class CurriculumStage(NamedTuple):
    """One phase of the rollout curriculum.

    ``rollout_hours``: autoregressive supervision horizon of the phase.
    ``n_epochs``: number of global epochs spent in this phase.
    ``lr_scale``: peak-LR multiplier for this phase (1.0 = the configured LR).
        Consumed by the piecewise optimizer schedule; the legacy tuple form
        always maps to 1.0.
    ``pushforward_no_grad_steps``: number of leading rollout steps run under
        ``stop_gradient`` (Brandstetter pushforward). 0 = full BPTT.
    """

    rollout_hours: float
    n_epochs: int
    lr_scale: float = 1.0
    pushforward_no_grad_steps: int = 0


def parse_curriculum(
    spec,
    *,
    allow_non_monotonic: bool = False,
) -> tuple[CurriculumStage, ...]:
    """Parse a curriculum spec into ``CurriculumStage`` tuples.

    Accepted forms (all produce the same structure):
    - falsy (``None``, ``()``, ``[]``) -> ``()`` (no curriculum);
    - legacy sequence of ``(lead_hours, n_epochs)`` pairs — the existing
      ``aimip_rollout_curriculum`` / ``NeuralGCMSpectralConfig.rollout_curriculum``
      YAML form ``[[12, 2], [24, 2], ...]``;
    - sequence of mappings ``{"rollout_hours": .., "n_epochs": ..,
      "lr_scale": .., "pushforward_no_grad_steps": ..}`` (extended form);
    - mapping ``{"stages": [...]}`` wrapping either of the above.

    Raises ``ValueError`` on malformed entries, non-positive hours/epochs,
    non-positive ``lr_scale``, negative pushforward, or a strictly decreasing
    ``rollout_hours`` ladder (short-lead-first doctrine) unless
    ``allow_non_monotonic=True``.
    """
    if not spec:
        return ()
    if isinstance(spec, dict):
        if "stages" not in spec:
            raise ValueError(
                "Curriculum dict form must be {'stages': [...]}; got keys "
                f"{sorted(spec)}."
            )
        spec = spec["stages"]

    stages: list[CurriculumStage] = []
    for entry in spec:
        if isinstance(entry, dict):
            unknown = set(entry) - set(CurriculumStage._fields)
            if unknown:
                raise ValueError(
                    f"Unknown curriculum stage keys {sorted(unknown)}; "
                    f"allowed: {list(CurriculumStage._fields)}."
                )
            try:
                stage = CurriculumStage(
                    rollout_hours=float(entry["rollout_hours"]),
                    n_epochs=int(entry["n_epochs"]),
                    lr_scale=float(entry.get("lr_scale", 1.0)),
                    pushforward_no_grad_steps=int(
                        entry.get("pushforward_no_grad_steps", 0)
                    ),
                )
            except KeyError as exc:
                raise ValueError(
                    f"Curriculum stage dict missing required key {exc}; "
                    "need at least 'rollout_hours' and 'n_epochs'."
                ) from None
        else:
            try:
                h, ep = entry
            except (TypeError, ValueError):
                raise ValueError(
                    "Legacy curriculum entries must be (lead_hours, n_epochs) "
                    f"pairs; got {entry!r}."
                ) from None
            stage = CurriculumStage(rollout_hours=float(h), n_epochs=int(ep))
        if stage.rollout_hours <= 0:
            raise ValueError(
                f"Curriculum stage rollout_hours must be > 0; got "
                f"{stage.rollout_hours}."
            )
        if stage.n_epochs < 0:
            raise ValueError(
                f"Curriculum stage n_epochs must be >= 0; got {stage.n_epochs}."
            )
        if stage.n_epochs == 0:
            # Legacy inline behaviour: a zero-epoch phase contributes nothing
            # to the epoch plan — skip it rather than erroring (a sweep
            # config may zero-out a phase to disable it).
            continue
        if stage.lr_scale <= 0:
            raise ValueError(
                f"Curriculum stage lr_scale must be > 0; got {stage.lr_scale}."
            )
        if stage.pushforward_no_grad_steps < 0:
            raise ValueError(
                "Curriculum stage pushforward_no_grad_steps must be >= 0; got "
                f"{stage.pushforward_no_grad_steps}."
            )
        stages.append(stage)

    hours = [s.rollout_hours for s in stages]
    if not allow_non_monotonic and any(
        b < a for a, b in zip(hours, hours[1:])
    ):
        raise ValueError(
            f"Curriculum rollout_hours ladder decreases ({hours}); the "
            "short-lead-first doctrine wants a non-decreasing ladder. Pass "
            "allow_non_monotonic=True if this is intentional."
        )
    return tuple(stages)


def total_epochs(stages: Sequence[CurriculumStage]) -> int:
    """Sum of ``n_epochs`` across stages (0 for an empty curriculum)."""
    return sum(s.n_epochs for s in stages)


def stage_for_epoch(
    stages: Sequence[CurriculumStage], epoch: int
) -> tuple[int, CurriculumStage]:
    """Return ``(stage_index, stage)`` owning global ``epoch`` (0-based).

    Raises ``ValueError`` when ``epoch`` is negative or past the last stage.
    """
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0; got {epoch}.")
    first = 0
    for i, s in enumerate(stages):
        if epoch < first + s.n_epochs:
            return i, s
        first += s.n_epochs
    raise ValueError(
        f"epoch {epoch} is past the curriculum end "
        f"(total_epochs={total_epochs(stages)})."
    )


def stage_boundaries(stages: Sequence[CurriculumStage]) -> list[int]:
    """First global epoch of each stage, e.g. ``[0, 2, 4, 6]``.

    Feed ``[b * steps_per_epoch for b in stage_boundaries(...)]`` to the
    piecewise optimizer schedule to restart warmup+decay per stage.
    """
    out, first = [], 0
    for s in stages:
        out.append(first)
        first += s.n_epochs
    return out


def build_curriculum_epoch_plan(
    curriculum,
    multi_step_hours,
    dt: float,
    n_epochs_fallback: int,
    *,
    require_exact: bool = False,
    dt_name: str = "dt",
):
    """Flat per-epoch plan ``[(lead_hours, k_target, n_steps), ...]``.

    THE shared implementation behind both the dycore-mode curriculum
    (``neural_gcm_spectral._train_spectral_loop``, micro-step ``config.dt``,
    permissive ``round``) and the sfno_full macro-step curriculum
    (``dt_sfno``, ``require_exact=True`` — a state-update SFNO advances in
    whole macro steps).

    ``curriculum`` may be anything :func:`parse_curriculum` accepts (already
    parsed stages included).  Falsy -> ``[(None, None, None)] * n_epochs_fallback``
    so the epoch loop indexes a uniform plan.  Each phase ``(h, ep)``
    contributes ``ep`` copies of ``(h, multi_step_hours.index(h),
    n_steps)``; resume (``start_epoch``) indexes into this list.

    Raises ``ValueError`` when ``multi_step_hours`` is empty while a
    curriculum is given, when a lead has no loaded target, or (with
    ``require_exact``) when a lead is not a positive exact multiple of ``dt``.
    Error messages are kept byte-compatible with the pre-extraction inline
    implementations.
    """
    # Permissive parse: the legacy call sites (dycore + sfno_full loops)
    # allow a phase order that revisits shorter leads (k_target is an index
    # into multi_step_hours, not the phase order), so the ladder-monotonic
    # doctrine is enforced only where suites are validated (campaign driver
    # / parse_curriculum default), never at plan-build time.
    if isinstance(curriculum, dict):
        # Dict-stage form ({"stages": [...]}) — parse straight through (it is
        # not indexable, and a pre-parsed sequence is never a dict).
        stages = parse_curriculum(curriculum, allow_non_monotonic=True)
    else:
        if curriculum is not None:
            # Materialize first: a generator must not lose its first item to
            # the CurriculumStage type probe below.
            curriculum = tuple(curriculum)
        stages = (
            curriculum
            if curriculum and isinstance(curriculum[0], CurriculumStage)
            else parse_curriculum(curriculum, allow_non_monotonic=True)
        )
    if not stages:
        return [(None, None, None)] * int(n_epochs_fallback)

    leads_loaded = tuple(int(h) for h in (multi_step_hours or ()))
    if not leads_loaded:
        raise ValueError(
            "rollout_curriculum needs loss_config.multi_step_hours to "
            "carry the curriculum leads (targets per lead)."
        )
    epoch_plan = []
    for s in stages:
        h = int(s.rollout_hours)
        if h not in leads_loaded:
            raise ValueError(
                f"Curriculum lead {h}h has no loaded target "
                f"(multi_step_hours={leads_loaded})."
            )
        steps_f = h * 3600.0 / dt
        n_steps = int(round(steps_f))
        if require_exact and (n_steps <= 0 or abs(steps_f - n_steps) > 1e-6):
            raise ValueError(
                f"Curriculum lead {h}h is not a positive exact multiple of "
                f"{dt_name}={dt:g}s ({steps_f:.4f} macro steps). Pick leads on "
                f"the {dt_name} grid (a state-update SFNO cannot take a "
                f"fractional final step)."
            )
        if s.pushforward_no_grad_steps >= max(n_steps, 1):
            raise ValueError(
                f"Curriculum stage lead {h}h: pushforward_no_grad_steps="
                f"{s.pushforward_no_grad_steps} must be < n_steps={n_steps} "
                "(at least one supervised step needs gradients)."
            )
        spec = (h, leads_loaded.index(h), n_steps)
        epoch_plan.extend([spec] * s.n_epochs)
    return epoch_plan
