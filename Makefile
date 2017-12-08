.PHONY: all
all: install

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: dev_install
dev_install: install
	pip install -r requirements-dev.txt

.PHONY: dev_install
test:
	py.test

.PHONY: ci
ci: test
	codecov --token e870ceb4-d7bf-49e4-bbae-105001b17231

.PHONY: format
format:
	find sca_tools tests -type f -name "*.py" ! -name "_version.py" \
		| xargs yapf --in-place

.PHONY: lint
lint:
	pylint -j4 sca_tools tests || true
	flake8 --exclude=_version.py sca_tools tests || true
	pydocstyle sca_tools tests || true

.PHONY: dist
dist:
	python setup.py sdist bdist_wheel
	twine upload dist/*
