#!/bin/sh

set -e

VAGRANT_TOX='/cygdrive/c/Python27/Scripts/tox --sitepackages'
VAGRANT_PYPATH='/cygdrive/c/Python26/:/cygdrive/c/Python27/:/cygdrive/c/Python33/:/cygdrive/c/Python34:/cygdrive/c/Python27/Scripts/'

vagrant up
vagrant rsync
vagrant ssh -c "cd /vagrant && PATH=$VAGRANT_PYPATH $VAGRANT_TOX"
