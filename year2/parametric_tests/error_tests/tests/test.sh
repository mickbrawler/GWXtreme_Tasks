#!/bin/bash
cores=20
samples=50
new=True
for (( c=1; c<=$cores; c++))
do
	python3 test.py $samples $c $new
	echo $c
done
