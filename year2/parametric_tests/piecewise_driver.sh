#!/bin/bash

for i in {1..1000}
do
   python error_seeker.py 1000 200 False
   sleep .25
done
