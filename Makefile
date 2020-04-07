build_tools:
	cd tools && $(MAKE)

build:
	mkdir -p build
##it may fail
build_cmake: | build
	cd build && cmake ..

build_depend:
	cd src && ./configure --static --use-cuda=no --static-fst --fst-root=../tools/openfst
	cd src && $(MAKE) depend

build_src: build_depend
	cd src && $(MAKE)

build_openblass:
	cd tools && $(MAKE) openblas

build_nnet:
	cd src/nnet3bin && $(MAKE) nnet3-latgen-faster-parallel-pipe

build_lm:
	cd src/latbin && $(MAKE) lattice-lmrescore-kaldi-rnnlm-pruned-pipe

.PHONY: build_cmake
