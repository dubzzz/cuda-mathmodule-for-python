execute: clean
	python setup.py build_ext --inplace

clean:
	rm -f src/mathmodule.cpp src/mathmodule.so src/checks/*.o src/kernels/*.o src/objects/*.o

