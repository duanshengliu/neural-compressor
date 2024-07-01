bench_layer:
	 OMP_NUM_THREADS=12 numactl -l -C 0-11  python  ./test/3x/torch/quantization/weight_only/layer_demo.py