#!/bin/bash
cores=32
samples=50000

rm -rf files/runs/*

for (( c=1; c<=$cores; c++))
do
	mkdir files/runs/core$c/
	python3 driver.py $samples $c &
done
