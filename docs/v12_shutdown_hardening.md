# V12 shutdown hardening

This corrects resource ownership after the Phase-2 historical checkpoint. It
does not alter Phase-1 quantity bounds, replay/fill rules, engineering duration
limits, historical tags or artifacts. No acquisition or evaluation is authorized.

Shutdown cancels and drains owned ingestion tasks, waits for shielded public
HTTP workers to finish their response contexts, attempts SESSION_CLOSE, and
attempts both journal closes independently. Startup failures after journal
creation use the same journal cleanup. Failed acknowledgement-file creation
closes the already-open event file. Already-closed handles are skipped; a handle
whose close failed can be retried. Cleanup does not rewrite session bytes.

Original exception objects and tracebacks remain visible. One failure is raised
unchanged; multiple failures use Python 3.12 BaseExceptionGroup (ExceptionGroup
when all members are Exceptions), with the original operation first and labeled
secondary errors after it. Cancellation/interrupt remains a failure. A second
cancellation cannot abandon the shielded shutdown task. Journal write/flush/fsync
errors propagate instead of being treated as reconnectable network outages.

The successful closure summary is written only after all owned cleanup succeeds.
Shutdown failure instead attempts an exclusive closure.failure.json sidecar with
FAILED/eligible=false and the failure tree. Failure to persist that sidecar is
also raised; a missing successful closure still rejects the session. Failed or
missing collector closure evidence must never be promoted to eligibility merely
because a separate raw replay can validate some bytes. Raw replay certification
is not a successful shutdown seal or research authorization. Missing acknowledgements
and truncated records remain detectable by the unchanged replay loader.

A restart must use a new directory/session identity. Existing directories are
rejected, preserving failed session history. Retrying cleanup cannot fabricate
a closure summary, clear a failure sidecar, or make a failed session successful.
