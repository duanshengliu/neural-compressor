bench_layer:
	 OMP_NUM_THREADS=8 numactl -l -C 0-7  python  ./test/3x/torch/quantization/weight_only/layer_demo.py