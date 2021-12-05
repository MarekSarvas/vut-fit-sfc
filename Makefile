SHELL := /bin/bash

install:
	bash install.sh

points_exp1: 
	source path.sh && cd src && python3 fuzzy_cmeans.py --task points --data_path ../data/points/data_9c_200_mu_11_var_2.csv --num_clusters 7 --epochs 30 --q 2 --save_img cmeans_points
	
covid:
	source path.sh && cd src && python3 fuzzy_cmeans.py --task img --data_path ../data/img/covid_01.jpeg --num_clusters 11 --epochs 30 --q 2 --save_img cmeans_covid

normal:
	source path.sh && cd src && python3 fuzzy_cmeans.py --task img --data_path ../data/img/normal_01.jpeg --num_clusters 11 --epochs 30 --q 2 --save_img cmeans_normal

pneum:
	source path.sh && cd src && python3 fuzzy_cmeans.py --task img --data_path ../data/img/pneumonia.jpeg --num_clusters 11 --epochs 30 --q 2 --save_img cmeans_pneumonia

app:
	source path.sh && cd src && python3 app.py --data_path ../data/points/data_9c_200_mu_11_var_2.csv --num_clusters 9 --epochs 30 --q 2 --data_path_img ../data/img/covid_01.jpeg --num_clusters_img 11 --epochs_img 40 --q_img 2 

app_default:
	source path.sh && cd src && python3 app.py 

zip:
	zip -r xsarva00.zip data/ SFC_documentation.pdf src/*.py Makefile install.sh path.sh requirements.txt 