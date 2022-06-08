#!/bin/bash
cores=20
samples=5000
# WARNING: Now realized this new variable only works if we use the same # of 
# cores everytime. When shifting with different core numbers it doesn't work. 
# Need to have combiner code be able to take in old data before collecting new
# data. Else this won't work. Need to transition to this bash driver clearing
# files/runs/ before running anything. This also means the new variable doesn't 
# matter. We'll always make new directories for each core for each run.
for (( c=1; c<=$cores; c++))
do
	python3 driver.py $samples $c &
done
