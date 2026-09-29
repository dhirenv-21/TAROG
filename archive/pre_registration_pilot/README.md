# Archived: pre-registration pilot files (obsolete, never used for results)

`run_pilot.py` and `pilot_fixtures.py` were written on 2026-09-28 (01:36–01:38 IST), *before*
the Phase 3 pre-registration (`d8728b3`). They were an ad-hoc go/no-go runner with its own
unregistered H1 assessment. They were superseded by `scripts/phase3/` and
`taorg.experiments.phase3`.

- No code or test imports them.
- No output of them exists (`outputs/pilot/` was never created), and no reported number comes
  from them.
- They are archived here, rather than deleted, to preserve the audit trail. They are not
  maintained and are not collected by pytest. `run_pilot.py` imports `tests.fixtures.pilot_fixtures`,
  which no longer exists at that path, so it will not run as-is.

Decision recorded in `research/phase4_preregistration.md` §12.
