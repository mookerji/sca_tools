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
