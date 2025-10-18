
import numpy as np
import random

def build_ngram(nsym, n, pp=None, sd=0):
    """
    Construct a hierarchical probabilistic n-gram model.

    Builds n-gram probability structures for symbols from 0 to `nsym - 1`
    up to order `n`. Each n-gram stores its probability, normalized weights,
    and child links to longer grams.

    Parameters
    ----------
    nsym : int
        Number of distinct symbols in the alphabet.
    n : int
        Maximum n-gram order (e.g., 3 for trigrams).
    pp : list[list[float]], optional
        Optional nested list of probabilities, where `pp[i][j]` specifies
        the probability for symbol `j` at order `i+1`. If None, uniform
        probabilities are used.
    sd : float, optional
        Standard deviation for Gaussian noise applied to probabilities
        (as a fraction of the mean). If 0, probabilities are deterministic.

    Returns
    -------
    ngrams : list[list[tuple[np.ndarray, float, float, list]]]
        Hierarchical n-gram structure.
        Each element represents an order of n-grams and contains tuples:
        `(a, p, p2, l)` where
            - `a` : np.ndarray
                The symbol sequence of length `i`.
            - `p` : float
                Normalized probability of this n-gram.
            - `p2` : float
                Combined probability from the previous order.
            - `l` : list
                List of child n-grams extending this one.

    Notes
    -----
    - The structure forms a tree, where each n-gram links to all possible
      (n+1)-grams beginning with it.
    - Probabilities are normalized within each order.

    Examples
    --------
    >>> ngrams = build_ngram(nsym=3, n=2)
    >>> len(ngrams[0])
    3
    >>> len(ngrams[1])
    9
    """

    def _prob(p, i, j):
        if pp is not None and i<len(pp) and j<len(pp[i]):
            p = pp[i][j]
        else:
            if p is None or p<0:
                p = 1/nsym
            if sd is not None and sd>0:
                p_ = 0
                n = 0
                while p_<=0: 
                    p_ = np.random.normal(p, p*sd, 1)
                    p_ = p_[0]
                    n += 1
                    if n==100:
                        p_ = p
                        break
                p = p_
        return p
        
    ngrams = []
    for i in range(1, n+1):
        igrams = []
        if i==1:
            w = 0
            for j in range(0, nsym):
                p = _prob(None, i, j)
                a = np.array([j], dtype=int)
                igrams.append((a, p, p, []))
                w += p
            for k in range(0, len(igrams)):
                (a, p, p2, l) = igrams[k]
                igrams[k] = (a, p/w, p2/w, l)
        else:
            grams_ = ngrams[i-2]
            for k in range(0, len(grams_)):
                a_,p_,p2_,l_ = grams_[k]
                w = 0
                m = len(igrams)
                for j in range(0, nsym):
                    a = np.zeros(i, dtype=int)
                    a[0:len(a_)]= a_
                    a[-1] = j
                    p = _prob(None, i, j)
                    p2 = -1
                    #print(f'{i}:', k, j, a_, p_, a, p, p2)
                    e = (a, p, -1, [])
                    igrams.append(e)
                    w += p
                for j in range(0, nsym):
                    (a, p, _, l) = igrams[m+j]
                    e = (a, p/w, p/w*p2_, l)
                    igrams[m+j] = e
                    l_.append(e)
       
        #print(igrams)
        ngrams.append(igrams)
    return ngrams

def ngram_find(gram, ngrams, fallback=True):
    """
    Find a specific n-gram within a hierarchical n-gram structure.

    Parameters
    ----------
    gram : np.ndarray
        Array representing the target n-gram sequence (e.g., `[1, 2, 0]`).
    ngrams : list
        The hierarchical n-gram structure produced by `build_ngram()`.
    fallback : bool, optional
        If True, recursively search shorter suffixes of `gram` if no exact
        match is found. Default is True.

    Returns
    -------
    tuple or None
        Matching n-gram entry `(a, p, p2, l)` if found, otherwise None.

    Examples
    --------
    >>> ngrams = build_ngram(3, 2)
    >>> g = np.array([0, 1])
    >>> ngram_find(g, ngrams)
    (array([0, 1]), 0.111..., 0.333..., [...])
    """
    
    i = gram.shape[0]
    if i>len(ngrams):
        return None
    igrams = ngrams[i-1]
    for j in range(0, len(igrams)):
        a,p,p2,l = igrams[j]
        if (a==gram).sum()==len(a):
            return igrams[j]
    if fallback and len(gram)>1:
        gram = gram[1:]
        return ngram_find(gram, ngrams) 
    return None

def ngram_sample(ngrams, nsym, n):
    """
    Generate a random symbol sequence of length `n` from an n-gram model.

    Sampling proceeds sequentially, selecting symbols based on conditional
    probabilities stored in the hierarchical n-gram structure.

    Parameters
    ----------
    ngrams : list
        Hierarchical n-gram model as returned by `build_ngram()`.
    nsym : int
        Number of symbols in the alphabet (used for indexing).
    n : int
        Desired length of the output sequence.

    Returns
    -------
    np.ndarray
        Generated symbol sequence of length `n`, with integer values in
        `[0, nsym - 1]`.

    Notes
    -----
    - If no valid continuation is found during sampling, the function
      truncates the sequence and returns it early.
    - Probabilities are sampled using uniform random draws.

    Examples
    --------
    >>> ngrams = build_ngram(3, 2)
    >>> seq = ngram_sample(ngrams, 3, 10)
    >>> seq.shape
    (10,)
    """

    def _ngram_sample1(igrams):
        q = random.random()
        w = 0
        for j in range(0, len(igrams)):
            _,p,p2,_ = igrams[j]
            w += p
            #print(j, q, w, p, igrams[j][0])
            if q<=w:
                #print('!', j, q, w, p, igrams[j][0])
                return igrams[j]
        #sanity fallback
        j = random.randint(0, len(igrams))
        return igrams[j]
    
    ss = np.zeros(n, dtype=int)
    for i in range(0, n):
        grams_ = []
        w = 0
        i_ = min(i, len(ngrams))
        if i==0:
            igrams = ngrams[i_]
            gram = _ngram_sample1(igrams)
            a,_,_,l = gram
            ss[i] = a[-1]
        else:
            l = []
            for k in range(max(0,i-len(ngrams)),i):
                _gram = ss[k:i]
                gram = ngram_find(_gram, ngrams)
                if gram==None:
                    ss = ss[0:i] #sanity check
                    #print('?!', i, k, ss[0:i])
                    return ss
                _,_,_,l = gram
                if len(l)>0:
                    break
            #print('!', i, ss[0:i], _gram, gram[0], len(l))
            if len(l)==0:
                ss = ss[0:i]
                return ss
            gram_ = _ngram_sample1(l)
            a_,_,_,_ = gram_
            ss[i] = a_[-1]
            #print('!!', i, ss[0:i], gram[0], len(l), a_)
    return ss

def ngram_msample(ngrams, nsym, n, m=1, as_array=True):
    """
    Generate multiple random sequences from an n-gram model.

    Parameters
    ----------
    ngrams : list
        Hierarchical n-gram model as returned by `build_ngram()`.
    nsym : int
        Number of symbols in the alphabet.
    n : int
        Length of each generated sequence.
    m : int, optional
        Number of sequences to generate. Default is 1.
    as_array : bool, optional
        If True, return results stacked in a 2D NumPy array.
        If False, return a list of 1D arrays.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Generated sequences. Shape `(m, n)` if `as_array=True`.

    Examples
    --------
    >>> ngrams = build_ngram(3, 2)
    >>> seqs = ngram_msample(ngrams, 3, 5, m=3)
    >>> seqs.shape
    (3, 5)
    """
    
    ss = []
    for i in range(0, m):
        s = ngram_sample(ngrams, nsym, n)
        ss.append(s)
    if as_array:
        return np.vstack(ss)
    return ss
        
        
def print_ngrams(ngrams, flat=True):
    """
    Print a human-readable summary of an n-gram model.

    Parameters
    ----------
    ngrams : list
        Hierarchical n-gram model from `build_ngram()`.
    flat : bool, optional
        If True, prints all n-grams at each order in a flat list.
        If False, prints the recursive structure as a tree.

    Returns
    -------
    None
        Outputs formatted text to stdout.

    Examples
    --------
    >>> ngrams = build_ngram(3, 2)
    >>> print_ngrams(ngrams)
    0: (#3) p:1.000 p2:1.000 3 1
      [0] : p:0.333 p2:0.333 l:3
      ...
    """
    
    nsym =  len(ngrams[0])

    def _print_ngrams(igrams, i):
        for j in range(0, len(igrams)):
            a,p,p2,l = igrams[j]
            tab = '  '*i
            print(f'{tab}{a} : p:{p:.3f} p2:{p2:.3f} l:{len(l)}')
            _print_ngrams(l, i+1)
            
    if flat:
        for i in range(0, len(ngrams)):
            igrams = ngrams[i]
            w = 0
            w2 = 0
            for j in range(0, len(igrams)):
                _,p,p2,_ = igrams[j]
                w += p
                w2 += p2
            if i>0:
                w /= nsym**i
            print(f'{i}: (#{len(igrams)}) p:{w:.3f} p2:{w2:.3f}', nsym, nsym**i)
            for j in range(0, len(igrams)):
                a,p,p2,l = igrams[j]
                print(f'  {a} : p:{p:.3f} p2:{p2:.3f} l:{len(l)}')
    else:
         _print_ngrams(ngrams[0], 0)
  
