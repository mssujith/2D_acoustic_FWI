#!/bin/bash
while true
do
	printf "JOBS \r"
	squeue -la
	sleep 10 
done
