.PHONY: all
all: install

.PHONY: install
install:
	pip install -r requirements.txt
	pip install codecov

.PHONY: test
test:
	py.test -v --cov=sca_tools tests/

.PHONY: ci
ci: test
	codecov --token e870ceb4-d7bf-49e4-bbae-105001b17231

.PHONY: format
format:
	find . -type f -name "*.py" | xargs yapf --in-place

.PHONY: dist
dist:
	pip install 'twine>=1.9.1' > /dev/null
	python setup.py sdist bdist_wheel
	twine upload dist/*
