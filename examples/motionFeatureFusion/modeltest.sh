#!/usr/bin/env sh
./build/tools/caffe train -gpu 0 -solver=examples/motionFeatureFusion/Demo/solver.prototxt -snapshot examples/motionFeatureFusion/Demo/snapshot/snap__iter_1600.solverstate