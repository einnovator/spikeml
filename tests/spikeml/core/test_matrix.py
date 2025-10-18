import numpy as np

from spikeml.core.matrix import matrix_split, normalize_matrix, _mult, cmask, cmask2, matrix_init, matrix_init2

def test_matrix_init():
    params = ConnectorParams(size=3)
    M = matrix_init(params)
    print('M:', M)

def test_normalize_matrix():
    M = np.array([[2.5 , 1], [1 , .5 ]])
    print(M)
    print(normalize_matrix(M, c_in=2, c_out=2, strict=False))
    print('-'*10)
    M = np.array([[2.5 , -1], [1 , .5 ]])
    print(M)
    print(normalize_matrix(M, c_in=2, c_out=2, strict=False))
    print('-'*10)
    print('-'*10)
    M = np.array([[2.5 , -2.5], [1 , -2 ]])
    print(M)
    print(normalize_matrix(M, c_in=2, c_out=2, strict=False))
    print('-'*10)
    M = np.array([[1 , 0], [.2 , 0 ]])
    print(M)
    print(normalize_matrix(M, c_in=0, c_out=1, strict=False))
    print('-'*10)

def test_cmask():
    M = np.array([[.7,0], [.3, 0]])
    M = np.array([[.8,.2], [.3, 0]])
    #print('M:', M)
    c_in,c_out=1,1
    d,d_in,d_out=cmask(M, c_in, c_out)
    xdisplay(M, d_out, d_in, d)
    M = np.array([[.8,.2], [.3, 0]])
    d,d_in,d_out=cmask2(M, c_in, c_out)
    xdisplay(M, d_out, d_in, d)

   
def test_matrix_init():
    params = ConnectorParams(size=3)
    M = matrix_init(params)
    print(f'{M}')

def test_matrix_init2():
    params = ConnectorParams(size=3)
    Mp,Mn = matrix_init2(params)
    print(f'{Mp}\n{Mn}')
    Mp,Mn = matrix_init2( ConnectorParams(size=3, mean=.2, sd=0), ConnectorParams(size=3, mean=.1, sd=0))
    print(f'{Mp}\n{Mn}')


if __name__ == '__main__':  
    test_matrix_init()
    test_normalize_matrix()
    test_cmask()
    
    test_matrix_init()
    test_matrix_init2()
