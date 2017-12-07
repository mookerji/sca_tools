.PHONY: run
run:
	bash ./cpu.sh

.PHONY: test
test:
	cat example.txt | python parse_sysbench.py
	python parse_sysbench.py example.txt
