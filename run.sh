#!/usr/bin/env sh

# Data preparation
./run_preprocessing.sh

# Evaluation of baseline models
./run_baselines.sh

# Training, Evaluation, Analysis
./run_dmv.sh
