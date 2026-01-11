from spikeml.datasets.encoder import one_hot

def test_one_hot():
    yy = np.array([i for i in range(4)])
    yy_ = one_hot(yy, yy.shape[0])
    print(yy_)