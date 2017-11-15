.PHONY: all
all: install

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: test
test:
	py.test -v tests/

.PHONY: ci
ci: test

.PHONY: format
format:
	find . -type f -name "*.py" | xargs yapf --in-place

.PHONY: dist
dist:
	python setup.py sdist upload -r pypi
