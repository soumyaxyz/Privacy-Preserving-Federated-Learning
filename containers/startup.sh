#!/bin/bash

# Copy everything to the workspace directory
cp -a . ../workspace

# Change directory to workspace
cd ../workspace/

# Run your scripts
./create_job.sh
./run_simulation.sh
