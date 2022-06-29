all: signals background

signals:
	python generate.py --file 1100 --signal 1
	python generate.py --file 1200 --signal 1
	python generate.py --file 1400 --signal 1
	python generate.py --file 1100 --signal 1 --tag r9364
	python generate.py --file 1200 --signal 1 --tag r9364
	python generate.py --file 1400 --signal 1 --tag r9364

background:
	python generate.py --file 5 --signal 0
	python generate.py --file 5 --signal 0 --tag r9364
