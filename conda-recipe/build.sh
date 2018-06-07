#!/bin/bash
$PYTHON setup.py install --single-version-externally-managed --record=record.txt

mkdir $PREFIX/photodiag-apps
cp -r apps/* $PREFIX/photodiag-apps
