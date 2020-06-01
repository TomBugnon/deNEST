src = denest
test = test

.PHONY: watch
watch: test
	watchmedo shell-command \
		--command='make test' \
		--recursive --drop --ignore-directories \
		--patterns="*.py" $(src) $(test)

.PHONY: test
test:
	python -m pytest --cov=denest test -v