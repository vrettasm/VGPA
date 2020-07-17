
class GaussianMoments(object):
    # ...
    __slots__ = ("m_arr", "v_arr")

    def __init__(self, m_arr, v_arr):
        self.m_arr = m_arr
        self.v_arr = v_arr
    # _end_def_

    def __call__(self, order=0):
        pass
    # _end_def_

    def dM(self, order=0):
        pass
    # _end_def_

    def dS(self, order=0):
        pass
    # _end_def_

# _end_class_
