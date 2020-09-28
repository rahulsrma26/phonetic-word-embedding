build:
	cd wsim && python setup.py build_ext --inplace

clean:
	rm -rf wsim/build
	rm wsim/wsim_wrapper.cpp
	rm wsim/wsim.cpython*.so

test:
	python test.py

train:
	python train.py ../data/cmudict-0.7b-with-vitz-nonce

reset:
	rm -rf checkpoints/
	rm -rf model.tf/
