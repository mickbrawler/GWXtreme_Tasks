#!/bin/bash
cores=64
samples=100000

rm -rf files/runs/*

for (( c=1; c<=$cores; c++))
do
	mkdir files/runs/core$c/
	python3 driver.py $samples $c &
done
