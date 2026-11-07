# momentum: `make reproduce` builds a local venv on first use, downloads the Ken French data, then runs every number and chart.
VENV   ?= .venv
DOCKER ?= docker

ifdef IN_DOCKER
PY    := python
STAMP :=
else
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.installed
endif

.PHONY: reproduce data docker clean

reproduce: $(STAMP)
	@$(PY) momentum.py
	@$(PY) walk_forward.py

# Re-download the current Ken French vintage (it is rebuilt monthly; numbers may move a little)
data: $(STAMP)
	@$(PY) data.py --refresh

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --disable-pip-version-check -r requirements.txt
	@touch $@

# Run `make reproduce` in a clean container: no local Python needed. DOCKER=podman works too.
docker:
	$(DOCKER) build -t momentum-after-the-paper .
	$(DOCKER) run --rm -v "$$PWD/results:/app/results" momentum-after-the-paper

clean:
	rm -rf results
