bench_layer:
	 USE_NUMBA=1 OMP_NUM_THREADS=6 taskset -c 0-5  python  ./examples/layer_demo.py


use_reshape:
	 USE_RESHAPE=1 OMP_NUM_THREADS=6 taskset -c 0-5  python  ./examples/layer_demo.py 
	 
use_reshape_numba:
	 USE_NUMBA=1 USE_RESHAPE=1 OMP_NUM_THREADS=6 taskset -c 0-5  python  ./examples/layer_demo.py 
	 