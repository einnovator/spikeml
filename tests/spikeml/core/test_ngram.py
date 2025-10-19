import numpy as np

from spikeml.core.ngram import build_ngram, ngram_find, ngram_msample, print_ngrams

def test_ngram():
    NSYM=2
    NGRAM=3
    ngrams = build_ngram(nsym=NSYM, n=NGRAM, sd=1)
    #print(ngrams)
    print_ngrams(ngrams, flat=True)
    print_ngrams(ngrams, flat=False)

    a = ngram_find(np.array([2,0]), ngrams)
    print(a)

    SLEN=10
    NS=6
    ss = ngram_msample(ngrams, nsym=NSYM, n=SLEN, m=NS) 
    print(ss)
    
          
