from spikeml.core.signal import signal_dc, signal_pulse, encode1_onehot, encode_onehot, signal_ranges, mean_per_input

def test_signal():
    ss = signal_dc(2, T=10, s=0, value=2)
    print(ss.tolist())
    ss = signal_pulse(2, T=5, L=2, s=[0,-1], value=1)
    print(ss.tolist())
    ss = signal_pulse(3, T=5, L=2, s=[0,1,2,-1], value=1)
    print(ss.tolist())

def test_signal2():
    s0 = np.array([1, 1, 0])
    ss = signal_pulse(3, T=3, L=1, s=[s0,-1,2,-1], value=1)
    print(ss.tolist())

def test_encode_onehot():
    encode1_onehot(0, 5)
    ss = signal_pulse(3, T=5, L=2, s=[0,1,2,-1], value=1)
    ess = encode_onehot(ss, NSYM)
    print(ess[0])
    


def test_signal_unique():
    ss = signal_pulse(2, T=5, L=2, s=[0,1,-1], value=1)
    print(ss.tolist())
    plot_xt(ss)
    print(signal_changes(ss))
    u = signal_unique(ss)
    print(u)
    ranges = signal_ranges(ss, ref=u, E=0)
    #print(ranges)
    for i,s in enumerate(u):
        print(i, ':', u[i], ranges[i])
        
def test_mean_per_input():
    ref, size, means = mean_per_input(monitor.err, nn.monitor.sx)
    for i in range(0, ref.shape[0]):
        print(f'{i}: {ref[i]} (#{size[i]}); Err: {means[i]:.4f}')




if __name__ == '__main__':  
    test_signal()
    test_signal2()
    test_encode_onehot()
    test_signal_unique()
    test_mean_per_input()