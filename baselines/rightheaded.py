class RightHeaded(object):

    def __init__(self):
        pass

    def parse(self, postags):
        """
        :type postags: list of str
        :rtype: list of (int, int)
        """
        assert postags[0] == "<root>" # NOTE
        arcs = []
        N = len(postags)
        for i in range(N-1, 1, -1):
            arcs.append((i, i-1)) # x_{i-1} <- x_{i}
        arcs.append((0, N-1))
        return arcs

