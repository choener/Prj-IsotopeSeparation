#!/usr/bin/env python3

# Small tool that generates a read statistics CSV for each fast5 read. These CSV are then consumed
# by the main isosep.py program.

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--outfile', help='csv with the output data. defaults to input with csv suffix')
parser.add_argument('input', nargs=1)

args = parser.parse_args()

