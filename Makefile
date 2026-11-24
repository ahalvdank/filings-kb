# EDGAR_IDENTITY (and any other local settings) come from .env when it exists.
-include .env
export

.PHONY: install test ingest signals sample sample-refresh traces traces-from-cache trace-check \
	parity evals

install:
	python3.11 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

# Cache-only ingest of the pinned sample; add ARGS=--live to extract with your own key.
# Like sample and signals, it writes no tracked file: its report goes to data/local/ and
# its duration to data/run_log.json (both gitignored), so make traces can run after it.
ingest:
	.venv/bin/python -m filings_kb ingest $(ARGS)

signals:
	.venv/bin/python -m filings_kb signals

# Full sample build, then the YoY signals. Extraction calls the model only for a section
# with no cached envelope; on a clean clone every section has one, so this re-uses the
# committed extractions and costs nothing. Use sample-refresh to re-extract. Signals go to
# data/local/signals.json, with any drift from data/committed/signals.json printed.
sample:
	.venv/bin/python -m filings_kb ingest --live
	.venv/bin/python -m filings_kb signals

# Re-extracts every detected section with your own key (ANTHROPIC_API_KEY), ignoring and
# then overwriting data/committed/extractions. Needs a fresh store: move data/kb.sqlite aside.
# The only target that rewrites the tracked data/committed/ outputs (--write-committed).
sample-refresh:
	.venv/bin/python -m filings_kb ingest --refresh --write-committed
	.venv/bin/python -m filings_kb signals --write-committed

# Answers every question with the claude CLI and exports exports/. Needs a clean git tree.
# Exits 1 when a question failed that the previous exports/failed.json did not list
# (failures are listed in exports/failed.json; raw answers in data/rejected/).
traces:
	.venv/bin/python -m filings_kb traces

# Re-exports the committed answers: no model call, no kb.sqlite, build_commit and failure
# reasons kept from the committed exports. Exits 0 even when exports/failed.json lists questions.
traces-from-cache:
	.venv/bin/python -m filings_kb traces --from-cache

# Offline trace check: citations resolve, numbers are in the cited passages. Writes data/audit_list.local.md
# (gitignored); only a run with ARGS=--network writes the committed exports/audit_list.md.
# Add ARGS=--network to also request every sec_url (needs EDGAR_IDENTITY in .env).
trace-check:
	.venv/bin/python -m evals.trace_check exports/traces $(ARGS)

# Parity against the author's private KB (not reproducible from a clean clone).
# Needs data/kb.sqlite (make ingest) and PRIVATE_KB=path/to/knowledge_graph.db.
parity:
	.venv/bin/python -m evals.parity --private-kb $(PRIVATE_KB)

# Every eval that needs no model call: the offline trace check, then parity when
# PRIVATE_KB is set (the author's private KB; skipped otherwise).
evals: trace-check
ifneq ($(strip $(PRIVATE_KB)),)
	$(MAKE) parity
else
	@echo "parity skipped: set PRIVATE_KB=path/to/knowledge_graph.db (needs the author's private KB)"
endif
