#!/bin/bash
$PYTHON setup.py install --single-version-externally-managed --record=record.txt

mkdir $PREFIX/photdiag-apps
cp -r apps/* $PREFIX/photdiag-apps