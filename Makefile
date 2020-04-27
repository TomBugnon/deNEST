src = nets
test = test

IGNORE = '\033[0;30m'
GOOD = '\033[0;32m'
BAD = '\033[1;31m'
END = '\033[0;38m'

SUCCESS = "$(GOOD)\\nSuccess.$(END)\\n"
FAILURE = "$(BAD)\\n\\n--------- FAILURE! ---------\\n$(END)\\n"

.PHONY: watch
watch: test
	watchmedo shell-command \
		--command='make test' \
		--recursive --drop --ignore-directories \
		--patterns="*.py" $(src) $(test)

.PHONY: test
test:
	py.test --cov=nets test -v

# python -m nets params/tree_paths.yml -i input.npy -o output
# echo "$(IGNORE)"
# colordiff params_old.yml output/params.yaml && echo "$(SUCCESS)" || echo "$(FAILURE)"
