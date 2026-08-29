SHELL := /bin/bash

ROOT := $(CURDIR)
PYTHON ?= python
MODEL ?= $(ROOT)/models/openvla-7b
RESULTS := $(ROOT)/artifacts/results
OPENVLA_RESULTS := $(RESULTS)/openvla_7b

export LIBERO_CONFIG_PATH := $(ROOT)/configs/libero
export MUJOCO_GL := egl
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help doctor test smoke quick benchmark cache all setup

help:
	@echo "make doctor     Check environment, GPU, paths, and imports"
	@echo "make quick      Run unit tests and the LIBERO smoke test"
	@echo "make benchmark  Run the short OpenVLA trick benchmark"
	@echo "make cache      Run the short VLA-Cache reproduction"
	@echo "make all        Run quick + both model benchmarks"
	@echo "make setup      Create/repair the vla_tricks environment"

doctor:
	$(PYTHON) scripts/doctor.py

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) scripts/smoke_libero.py

quick: doctor test smoke

benchmark:
	mkdir -p $(OPENVLA_RESULTS)
	$(PYTHON) scripts/benchmark_openvla.py --model $(MODEL) --backend sdpa \
		--warmup 1 --repeats 3 --prune-layers 1 \
		--output $(OPENVLA_RESULTS)/dense_sdpa_diagnostics.json

cache:
	mkdir -p $(OPENVLA_RESULTS)
	$(PYTHON) scripts/benchmark_vla_cache.py --model $(MODEL) \
		--warmup 1 --repeats 3 --static-patches 130 \
		--output $(OPENVLA_RESULTS)/vla_cache_diagnostics.json

all: quick benchmark cache

setup:
	bash scripts/setup_env.sh
