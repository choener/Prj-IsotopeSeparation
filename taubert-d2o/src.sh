#!/usr/bin/env bash

# Link in sources depending on the network we are in.

H=`hostname -f`
D=`dirname $0`
cd $D

case $H in
  *.bioinf.uni-jena.de)
    echo "$H -> uni jena"
    ln -s /data/fass5/reads/heavy_atoms_deuterium_taubert/basecalled_fast5s/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing/ reads
    ;;
  yennefer)
    echo "$H -> home network"
    ln -s /shared/choener/Workdata/heavy_atoms_deuterium_taubert/20220303_FAR96927_BC14-15-16_0-30-100_Deutrium_sequencing/ reads
    ;;
  *)
    echo "$H -> unknown"
    echo "Host unknown, data is not automatically prepared"
    ;;
esac

