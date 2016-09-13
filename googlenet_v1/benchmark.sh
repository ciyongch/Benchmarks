#!/bin/sh
source ~/.bashrc

rm -r /tmp/wangfei/.theano/compile*/*

python googlenet_benchmark.py
