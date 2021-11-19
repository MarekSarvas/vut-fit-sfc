SHELL := /bin/bash

exp1:
	source path.sh && cd src && python3 fuzzy_cmeans.py --ep 200
	
exp2:
	source path.sh && cd src && python3 fuzzy_cmeans.py --ep 400 --data ../data/data_9c_200_mu_11_var_2.csv --clusters 5