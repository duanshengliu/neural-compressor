bench_layer:
	 OMP_NUM_THREADS=6 taskset -c 0-5  python  ./examples/layer_demo.py