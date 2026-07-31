# LAMP-Rec public interface skeleton. Requires a Python 3.11 venv (see README).
PY = uv run --directory . python
PYTEST = uv run --directory . --extra dev pytest

.PHONY: all test rq1 rq2 rq3 rq4 rq5 rq6 anchors clean

all:            ## run every RQ then rebuild the paper anchors/tables
	$(PY) experiments/run_all.py
	$(PY) experiments/report/build_anchors.py

test:           ## run the (public portion of the) unit tests
	$(PYTEST) -q

rq1:; $(PY) experiments/rq1_tracking.py
rq2:; $(PY) experiments/rq2_coverage.py
rq3:; $(PY) experiments/rq3_alarm.py
rq4:; $(PY) experiments/rq4_delay.py
rq5:; $(PY) experiments/rq5_ablation.py
rq6:; $(PY) experiments/rq6_robustness.py

anchors:        ## results/ -> paper anchors.json + regenerated tables
	$(PY) experiments/report/build_anchors.py

clean:          ## remove raw per-seed results (keep summaries)
	find results -name 'seed_*.json' -delete
