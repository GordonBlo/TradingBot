# V10 prospective overnight acquisition

Workflow: `.github/workflows/v10-microstructure-overnight.yml`.
Manual dispatch starts three sequential jobs on separate GitHub-hosted Ubuntu
runners. Each collects 10,800 seconds, with a 240-minute job timeout for setup,
closure, replay and upload. A repository-wide concurrency group prevents two
V10 overnight campaigns from running together, including across branches.

Each job verifies exact preregistration `8326791b411c27b5`, its file hash, frozen
collector source hashes and the strictly-after-cutoff rule before collection.
The canonical root must initially be absent. Its sole session is checked by
the frozen acquisition scanner after normal closure. One eligible session may
pass its job while the scanner's overall dataset status remains NOT_READY;
the 8-session / 24-hour / 3-start-date gate applies to the assembled dataset.

The four original artifacts remain unchanged. An exclusive-create
`acquisition.validation.json` records the preflight, complete scanner report,
artifact hashes, GitHub run/attempt and session result. Only these five files
are uploaded; the parent acquisition directory and other sessions are never
uploaded. Artifact names include run ID, attempt and session ID, with four-day
retention. Later import must verify all hashes and rerun the frozen scanner;
stored validation is provenance, not permission to bypass current integrity.

An identifiable failed or interrupted session can still be uploaded for
engineering inspection, with failure metadata. It contributes zero eligible
hours. Later sessions continue after earlier failures unless the campaign is
cancelled. The final summary also requires successful artifact upload before
reporting eligibility. Missing jobs or outputs are reported FAIL, not omitted.

Three successful campaigns whose sessions cover at least three UTC start dates
produce nine sessions and 27 eligible hours. Download the artifacts before they
expire. The campaign summary covers this run only; it is not a cross-run readiness
or import service. GitHub queue delays and UTC midnight can affect start dates.

This workflow collects acquisition evidence only. It performs no predictive
evaluation, authenticated requests or orders. The sessions may support later
authorized discovery; they are not independent confirmatory strategy evidence
unless a strategy hypothesis was frozen before session generation.

Once the files are on the repository default branch, open Actions, select
"V10 microstructure overnight prospective collection", choose Run workflow,
select the branch and confirm Run workflow. No schedule is installed.

ALWAYS use `Run workflow` for each new campaign, creating a NEW
`workflow_dispatch` run. NEVER use `Re-run all jobs` for a completed research
campaign. Attempts other than 1 fail before checkout, collection or artifact
upload in every job, including the summary job. The guard does not recover
artifacts already lost; it is not a substitute for downloading them.

Research campaigns must be started with a NEW workflow_dispatch run. Do not use Re-run all jobs because GitHub replaces artifacts from the previous attempt.

Download and import session artifacts before the four-day retention expiry.
