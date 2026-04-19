VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv notebook clean

help:
	@echo "Available targets:"
	@echo "  venv       Create virtualenv and install dependencies"
	@echo "  notebook   Launch Jupyter notebook server"
	@echo "  clean      Remove virtualenv"

venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

notebook: venv
	$(VENV)/bin/jupyter lab notebooks/

clean:
	rm -rf $(VENV)
