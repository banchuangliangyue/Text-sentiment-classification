import csv
import sys
import pandas as pd
import copy


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

input_file = 'sentiment-analysis-on-movie-reviews/train.tsv'
lines= _read_tsv(input_file)

print(lines[:10])
