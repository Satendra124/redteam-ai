#!/bin/bash

CHECKPOINT_DIR=/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-8B-Instruct
PYTHONPATH=$(git rev-parse --show-toplevel) torchrun main.py $CHECKPOINT_DIR