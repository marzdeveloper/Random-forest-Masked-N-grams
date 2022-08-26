#!/usr/bin/env python3 -u

import signal
import sys
from ngram import NGram
import itertools
import collections
import statistics

if len(sys.argv) == 4:
  tag = sys.argv[1]
  domain_file = sys.argv[2]
  Nsize = int(sys.argv[3])
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


# ID Designation
# F1-3   1-gram (mean, variance and standard deviation)
# F4-6   2-gram (mean, variance and standard deviation)
# F7-9   3-gram (mean, variance and standard deviation)
# F10-12 4-gram (mean, variance and standard deviation)
# F13-14 Shannon entropy in 2LD and 3LD
# F15    Number of different characters
# F16    Number of digits / domain name length
# F17    Number of consonants / domain name length
# F18    Number of consonants / number of vowels
# F19-21 TTL, Window (first seen - last seen), Ratio (window / count)
# F22-24 Number of different first seen, answers and TTL
# F25-27 Maximum Ratio, TTL and Window
# F28-30 Mean Ratio, TTL and Window
# F31-33 Variance of Ratio, TTL and Window
# F34-36 Number of IP subnetworks

with open(domain_file) as f:
  for line in f:
    domain_name = line.rstrip('\n')
    feature = []

    aux1_domain_name = multi_replace( domain_name, ['b', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'], "c" )
    aux2_domain_name = multi_replace( aux1_domain_name, ['a', 'e', 'i', 'o', 'u'], "v" )
    aux3_domain_name = multi_replace( aux2_domain_name, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], "n" )
    masked_domain_name = multi_replace( aux3_domain_name, ['-'], "s" )
    #print domain_name + " -> " + masked_domain_name
    #continue

    feature.append(domain_name)
    feature.append(masked_domain_name)
    feature.append(tag)

    if len(domain_name) < 5:
      continue

    ### ID Designation
    ### V4-6   1-gram (mean, variance and standard deviation)
    n = NGram(N=1)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.extend([f1, f2, f3])

    ### V6-9   2-gram (mean, variance and standard deviation)
    n = NGram(N=2)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.extend([f1, f2, f3])

    ### V10-12   3-gram (mean, variance and standard deviation)
    n = NGram(N=3)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.extend([f1, f2, f3])

    ### V13-15 4-gram (mean, variance and standard deviation)
    n = NGram(N=4)
    v = list(n._split(domain_name))
    [f1, f2, f3] = ngram_stats(v)
    feature.extend([f1, f2, f3])

    # V16    Number of different characters
    different_characters = len(set(list(domain_name)))
    feature.append( different_characters )

    # V17    Number of digits / domain name length
    domain_length = len(domain_name)
    number_of_digits = masked_domain_name.count('n')
    feature.append( number_of_digits / domain_length )

    # V18    Number of consonants / domain name length
    number_of_consonants = masked_domain_name.count('c')
    feature.append( number_of_consonants / domain_length )
    
    # V19    Number of consonants / number of vowels
    number_of_vowels = masked_domain_name.count('v')
    if number_of_vowels > 0:
      feature.append( number_of_consonants / number_of_vowels )
    else:
      feature.append( 0 )

    # NEW approach, using masked domain and full ngrams list

    # V20 Domain name length
    feature.append( domain_length )

    # V21-... N-Grams
    test_features_hash = {}
    for n in ["".join(p) for p in itertools.product(["c", "v", "n", "s"], repeat=Nsize)]:
      test_features_hash[n] = 0
    n = NGram(N=Nsize)
    v = list(n._split(masked_domain_name))
    for n in v:
      if n in test_features_hash.keys():
        test_features_hash[n] += 1
    feature.extend(test_features_hash.values())

    csv_features = ",".join(str(a) for a in feature)
    print(csv_features)

