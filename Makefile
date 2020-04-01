
build_tools:
	cd tools && $(MAKE)

##it may fail
build_cmake:
	cd build && cmake ..

build_src:
	cd src && ./configure --static --use-cuda=no --static-fst --fst-root=../tools/openfst
	cd src && $(MAKE) depend
	cd src && $(MAKE)

build_openblass:
	cd tools && $(MAKE) openblas

build_nnet:
	cd src/nnet3bin && $(MAKE) nnet3-latgen-faster-parallel-pipe

.PHONY: build_cmake
