PYTHON = python3
VENV = brainmap
ACTIVATE = $(VENV)/bin/activate
TESTS = tests


install:
	pip install -r requirements.txt
