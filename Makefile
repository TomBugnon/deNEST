.PHONY: watch test docs dist

src = denest
test = test
docs = docs
docs_source = docs/source
docs_build = docs/build
docs_html = docs/build/html
docs_port = 7331
dist_dir = dist

watch: test
	watchmedo shell-command \
		--command='make test' \
		--recursive --drop --ignore-directories \
		--patterns="*.py" $(src) $(test)

test:
	python -m pytest --cov=denest test -v

docs: build-docs

watch-docs: docs
	watchmedo shell-command \
		--command='make build-docs' \
		--recursive --drop --ignore-directories \
		--patterns="*.py;*.rst" $(src) $(docs)

build-docs:
	cd $(docs) && make html -j12
	# cp $(docs)/_static/*.css $(docs_html)/_static
	# cp $(docs)/_static/*.png $(docs_html)/_static

serve-docs: build-docs
	cd $(docs_html) && python -m http.server $(docs_port)

open-docs:
	open http://0.0.0.0:$(docs_port)

clean-docs:
	rm -rf $(docs_build)
	rm -rf $(docs_source)/generated/*

check-dist:
	python setup.py check --strict

dist: build-dist check-dist
	twine upload $(dist_dir)/*

test-dist: build-dist check-dist
	twine upload --repository-url https://test.pypi.org/legacy/ $(dist_dir)/*

build-dist: clean-dist
	python setup.py sdist bdist_wheel --dist-dir=$(dist_dir)

clean-dist:
	rm -rf $(dist_dir)

clean: clean-dist
	rm -rf **/__pycache__