class SSNN4(Module):
    """
    Dynamic Stochastic Rectified Linear Layer (aka. stochastic rectified linear state space model) 
    """

    def __init__(self, x0=None, A=None, B=None, C=None, D=None, stateful=True, vmin=VMIN, vmax=VMAX, monitor=None, name=None, callback=None):
        super().__init__(name=name, callback=callback, monitor=monitor)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        if x0 is None and A is not None:
            x0 = np.zeros(A.shape[1], dtype=float)
        self.x0 = x0
        self.x = copy.copy(x0)
        self.y = None
        self.s = None
        self.stateful = stateful
        self.vmin = vmin
        self.vmax = vmax

    def reset(self):
        self.x = self.x0
        
    def dump(self):
        s = self.__str()
        print(s)
        
    def __str(self):
        ss = []
        if self.A is not None:
            ss.append(f'{self.name}: A= {self.A}')
        if self.B is not None:
            ss.append(f'{self.name}: B= {self.B}')
        if self.C is not None:
            ss.append(f'{self.name}: C= {self.C}')
        if self.D is not None:
            ss.append(f'{self.name}: D= {self.D}')
        s = '\n'.join(ss)
        return s

    def propagate(self, s):
        x = self.x
        x = (self.A @ x if self.A is not None else x)
        s_ = self.B @ s if self.B is not None else s
        x = (x + s_) if x is not None else s_
        self._x = x
        if x is not None:
            x = np.clip(x, self.vmin, self.vmax)
        y = self.C @ x if self.C is not None else None
        if self.D is not None:
            if y is not None:
                y = y + self.D@s
            else:
                y = self.D@s
        self._y = y
        y = np.clip(y, self.vmin, self.vmax)
        self.y = y
        self.s = s
        if self.stateful:
            self.x = x
        print(f'{self.name}: #s: {s.shape} ; #y: {y.shape} | s= {s} -> y={y}')

        if ref.A is not None:
            ref.A.progagate(s)
        if ref.B is not None:
            ref.B.progagate(s)
        if ref.C is not None:
            ref.C.progagate(s)
        if ref.D is not None:
            ref.D.progagate(s)

        return y


class SSNN4Monitor(Monitor):
    def __init__(self):
        #super().__init__(name=name, callback=callback)
        pass
        self.x = []
        self.y = []
        self.s = []
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        
    def sample(self, ref):
        self.s.append(ref.s) 
        self.y.append(ref.y) 
        self.x.append(ref.x) 
        if ref.A is not None:
            if self.A is None:
                self.A = []
            self.A.append(ref.A)
        if ref.B is not None:
            if self.B is None:
                self.B = []
            self.B.append(ref.B)
        if ref.C is not None:
            if self.C is None:
                self.C = []
            self.C.append(ref.C)
        if ref.D is not None:
            if self.D is None:
                self.D = []
            self.D.append(ref.D)

            