# Dry-Run Registry

Every deterministic / regression-test scenario the pipeline owns lives as
`_dryrun_p198<id>_<topic>.py` at the repo root. They run pure-Python (no
LLM/VLM/server) so you can execute any one — or all of them — at any time
without touching the live system.

## How to run

```bash
# Run a single scenario (Windows console: prefix with PYTHONIOENCODING=utf-8
# if it uses unicode arrows in print statements)
PYTHONIOENCODING=utf-8 python _dryrun_p198dp_dr_guard.py

# Run ALL scenarios and get a one-line summary per script
python _run_all_dryruns.py
```

## Registry

| ID | File | Tests | Notes |
|----|------|-------|-------|
| P198da | [_dryrun_p198da_47a_evidence.py](_dryrun_p198da_47a_evidence.py) | F47A "needs evidence" recogniser (charges-on-DR + SWIFT advice clauses) — deterministic guard accuracy AND LLM agreement | Hits live LLM endpoint when network is up; deterministic guard alone is the regression bar |
| P198da-handled | [_dryrun_p198da_handled_guard.py](_dryrun_p198da_handled_guard.py) | `_p198da_handled` flag survives `_build_tasks` skip-task / P169 drop-from-report rules | Pure-Python |
| P198db | [_dryrun_p198db_swift_routing.py](_dryrun_p198db_swift_routing.py) | Non-amendment MT799/MT999 routes to BOTH `mt_packets` AND `shipping_packets` with `is_swift_advice_copy=True` | Pure-Python |
| P198dc | [_dryrun_p198dc_f45a_packinglist.py](_dryrun_p198dc_f45a_packinglist.py) | F45A goods-description / quantity rows clone to a Packing List row with `condition_id` suffix `-PL`; price / Incoterms rows are NOT cloned | Pure-Python |
| P198dd | [_dryrun_p198dd_expiry_bound.py](_dryrun_p198dd_expiry_bound.py) | F48 "BUT WITHIN EXPIRY" relaxes the 21-day late-presentation rule when the actual presentation is still ≤ F31D | Pure-Python |
| P198de | [_dryrun_p198de_f48_parser.py](_dryrun_p198de_f48_parser.py) | F48 day-count parser — accepts "21 DAYS", BAHL slash-form "15/FROM SHIPMENT", "21/FRM B/L", bare leading "30 FROM PRESENTATION" | 20 scenarios |
| P198df | [_dryrun_p198df_f48_display.py](_dryrun_p198df_f48_display.py) | Final-LC F48 display reformat — "15/FRM SHIPMENT DATE BUT WITH IN EXPIRY" → "15 days from shipment date but within expiry" | 15 scenarios |
| P198dg | [_dryrun_p198dg_report_text.py](_dryrun_p198dg_report_text.py) | Report has no LLM/VLM mentions in user-facing text + table-cell text not truncated | Pure-Python |
| P198dh | [_dryrun_p198dh_page_xy.py](_dryrun_p198dh_page_xy.py) | "Page X/Y" slash form is recognised alongside "Page X of Y" | 20 scenarios |
| P198di | [_dryrun_p198di_field_ref_resolver.py](_dryrun_p198di_field_ref_resolver.py) | F47A simple-ref resolver only fires on bare references; multi-clause F47A with "AS PER FIELD 45A" inside one clause is preserved | 8 scenarios incl. real job 08345848 |
| P198dj | [_dryrun_p198dj_strip_ucp.py](_dryrun_p198dj_strip_ucp.py) | UCP/ISBP citations stripped from user-facing findings (in-code source + stored result JSONs) | Sweeps `results/*` |
| P198dk | [_dryrun_p198dk_swift_advice_content.py](_dryrun_p198dk_swift_advice_content.py) | F47A SWIFT-advice content check — issuance-time advising-bank MT799 fails, post-negotiation MT799 with vessel/voyage/IMO/BL/etc. passes | 6 scenarios incl. job 08345848 pages 1+2 |
| P198dl | [_dryrun_p198dl_proforma_pl_opt.py](_dryrun_p198dl_proforma_pl_opt.py) | F45A proforma OPPORTUNISTIC clone to PL — silently skips when PL has no proforma reference, hands to LLM when PL carries one | 6 scenarios |
| P198dm | [_dryrun_p198dm_proforma_pl_postcheck.py](_dryrun_p198dm_proforma_pl_postcheck.py) | P198ak proforma date integrity post-check now also fires for Packing List rows; CI/PL scopes isolated so cross-doc mismatches don't leak | 7 scenarios incl. real job 08345848 |
| P198dn | [_dryrun_p198dn_no_table_truncation.py](_dryrun_p198dn_no_table_truncation.py) | No `[:200]` / `[:300]` / `[:500]` / `[:600]` truncations remain on row['result'] / row['found_text'] / parsed['result'] in step14, and step20 table cells use `max_len=100000` | Static code scan |
| P198dp | [_dryrun_p198dp_dr_guard.py](_dryrun_p198dp_dr_guard.py) | Documentary Remittance false-positive guard — real bank covering schedules KEEP, email cover notes / endorsement back pages / random docs DEMOTE | 15 scenarios (3 real job 406fec4f pages + 12 synthetic) |
| P198dp-real | [_dryrun_p198dp_dr_guard_real_jobs.py](_dryrun_p198dp_dr_guard_real_jobs.py) | Sweeps every DR-tagged packet in `results/*/step08/` and reports KEEP / DEMOTE counts | 49 packets across the local results store |
| P198dv | [_dryrun_p198dv_insurance_request_shipment_advice.py](_dryrun_p198dv_insurance_request_shipment_advice.py) | F46A clauses targeting "Shipment Advice" match pages classified as Insurance Request / Insurance Cover Request / Insurance Pre-Advise Notice / etc.; Insurance Policy clauses do NOT spuriously match these emails | 26 assertions incl. real job 38beca01 email body |

## Conventions

- All dry-runs are **pure-Python** (no live LLM/VLM/server) so they can be
  re-run any time without side effects. Exception: the P198da scenario
  optionally hits the live LLM for an "agreement" metric — the
  deterministic guard accuracy is what matters for regression.
- Every dry-run uses **real OCR / packet data** from a job in `results/`
  whenever possible, plus synthetic scenarios for edge cases.
- Pass criterion is printed at the end of each script as `OVERALL: N/N OK`
  or similar — the runner script (`_run_all_dryruns.py`) parses these.
- Add new dry-runs as `_dryrun_p198<id>_<topic>.py` and add a row to the
  table above so the runner picks it up.
