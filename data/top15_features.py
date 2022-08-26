#!/usr/bin/env python3 -u

import signal
import sys
from ngram import NGram
import itertools
import collections
import statistics

if len(sys.argv) == 3:
  tag = sys.argv[1]
  domain_file = sys.argv[2]
else:
  print('Invalid params!')
  sys.exit(0)

def signal_handler(signal, frame):
  print('Exiting!')
  sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def ngram_stats( vector ):
  counter=collections.Counter(vector)
  vector_values = counter.values()
  m = statistics.mean(vector_values)
  v = statistics.variance(vector_values)
  s = statistics.stdev(vector_values)
  return [m, v, s]

def multi_replace( source, chars_out, char_in ):
  for c in chars_out:
    source = source.replace(c, char_in)
  return source

with open(domain_file) as f:
  for line in f:
    domain_name = line.rstrip('\n')
    feature = []

    aux1_domain_name = multi_replace( domain_name, ['b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'], "c" )
    aux2_domain_name = multi_replace( aux1_domain_name, ['a', 'e', 'i', 'o', 'u'], "v" )
    aux3_domain_name = multi_replace( aux2_domain_name, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], "n" )
    masked_domain_name = multi_replace( aux3_domain_name, ['-'], "s" )

    feature.append(domain_name)
    feature.append(masked_domain_name)
    feature.append(tag)

    if len(domain_name) < 5:
      continue

    ### ID Designation
    # Features 1-3: 1-gram mean, variance and standard deviation
    n = NGram(N=1)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.extend([f1, f2, f3])

    # Feature 4: 2-gram standard deviation
    n = NGram(N=2)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.append(f3)

    # Feature 5: Number of different characters
    different_characters = len(set(list(domain_name)))
    feature.append( different_characters )

    # Feature 6: Domain name length
    domain_length = len(domain_name)
    feature.append( domain_length )

    # Feature 7-15: NGrams
    ngram_features = ["ccc", "cvc", "vcc", "vcv", "cv", "vc", "cc", "c", "v"]
    ngram_dict = {}
    for i in ngram_features:
        ngram_dict[i] = 0
    for i in [1, 2, 3]:
        ng = NGram(N=i)
        v = list(ng._split(masked_domain_name))
        for n in v:
            if n in ngram_features:
                ngram_dict[n] += 1
    feature.extend(ngram_dict.values())

    csv_features = ",".join(str(a) for a in feature)
    print(csv_features)

