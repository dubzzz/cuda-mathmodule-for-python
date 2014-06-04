CCFLAGS := 
ifeq ($(dbg),1)
	CCFLAGS += -g -D_DEBUG
endif

execute: clean
	python setup.py build_ext --inplace $(CCFLAGS)

clean:
	rm -f src/mathmodule.cpp src/mathmodule.so src/checks/*.o src/kernels/*.o src/objects/*.o

