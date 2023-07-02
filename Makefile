
PYTHON ?= python
PIP ?= pip

dist:
	$(PYTHON) -m build

.PHONY: dist

install:
	$(PIP) install .

test:
#	tox
	pytest tests

VENV = env
init-venv: $(VENV)

$(VENV):
#	virtualenv --system-site-packages $(VENV)
	virtualenv $(VENV)

clean-venv:
	rm -rf $(VENV)

install-pkgs: install-requirements

install-requirements: init-venv
	which pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf build dist

allclean: clean
	rm -rf .tox __pycache__ $(VENV)
