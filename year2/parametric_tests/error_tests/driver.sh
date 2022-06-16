#!/bin/bash
cores=2
samples=5000
spectral=0

rm -rf files/runs/*

for (( c=1; c<=$cores; c++))
do
	mkdir files/runs/core$c/
	python3 driver.py $samples $c $spectral &
done
