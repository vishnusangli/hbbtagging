clean:
	rm epochs/*.npy
	rm scores/*

request:
	salloc --signal=USR1@60 -N 1 -C haswell -q interactive -t 04:00:00
	conda activate myroot
