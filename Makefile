build-compute:
	cmake -S ./ -B build/ -DCMAKE_EXPORT_COMPILE_COMMANDS=1
clean:
	rm -rf build compile_commands.json
