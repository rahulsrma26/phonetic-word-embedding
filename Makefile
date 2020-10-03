CDIR := ${CURDIR}

build:
	docker build -t phonetics:latest .

develop:
	docker run --gpus 1 -it --rm -v $(CDIR):/workspace phonetics:latest

clean:
	docker rmi phonetics
	docker image prune -f

clean_base:
	docker rmi continuumio/miniconda3
	docker image prune -f
