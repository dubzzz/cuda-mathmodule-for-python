CCFLAGS := 
ifeq ($(dbg),1)
	CCFLAGS += -g -D_DEBUG
endif

all: test
rebuild: clean build

test: build
	python tests/benchmark.py
	python tests/test.py

build: clean
	python setup.py build_ext --inplace $(CCFLAGS)

clean:
	rm -f src/mathmodule.cpp src/mathmodule.so src/checks/*.o src/kernels/*.o src/objects/*.o
