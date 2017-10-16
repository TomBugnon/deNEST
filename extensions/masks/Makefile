build_dir = build
extension_module = RotatedRectangularMaskModule

build: clean
	mkdir -p ./$(build_dir)
	cd $(build_dir) \
		&& cmake ../$(extension_module) \
		&& make \
		&& make install

watch: build
	watchmedo shell-command \
		--command='make build' \
 		--recursive --drop --ignore-directories \
		--patterns="*.h;*.cpp" \
		$(extension_module)

clean:
	rm -rf ./$(build_dir)
