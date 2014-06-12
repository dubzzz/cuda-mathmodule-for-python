cuda-mathmodule-for-python
==========================

Python module for Linear Algebra computations using CUDA. The module relies on Cython and NumPy/C API.

Converting a NumPy array to a GPU-array:
```Python
import numpy as np, mathmodule as mm
numpy_array = np.random.random(5)
gpu_array = mm.PyVector(numpy_array)
```

And GPU-array to NumPy array:
```Python
numpy_array = gpu_array.toNumPy()
```
