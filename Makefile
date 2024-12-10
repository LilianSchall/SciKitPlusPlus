all: download-examples build-compute

download-examples:
	wget https://s3l4h.com/static/examples.zip && mv examples.zip tests && cd tests && unzip examples.zip && rm examples.zip

build-compute:
	cmake -S ./ -B build/ -DCMAKE_EXPORT_COMPILE_COMMANDS=1 && cp -r tests/examples build

clean:
	rm -rf build compile_commands.json tests/examples
