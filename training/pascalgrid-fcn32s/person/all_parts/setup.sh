#!/bin/bash

CAFFE_ROOT='/home/cv/hdl/caffe'
MODEL='pascalgrid-fcn32s'
OBJ='person'
PART='all_parts'

mkdir log

mkdir $CAFFE_ROOT/models/$MODEL/$OBJ/$PART/snapshot
ln -s $CAFFE_ROOT/models/$MODEL/$OBJ/$PART/snapshot snapshot

ln -s $CAFFE_ROOT/models/$MODEL/$OBJ/$PART/ model
python model/net.py
