import numpy as np
from pynndescent import NNDescent

# helper function
def hausdorff_d(curr_data, prev_data):
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((curr_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(curr_data.shape[0]))))
    # distance metric
    metric = "euclidean"
    # get nearest neighbors
    nnd = NNDescent(
        curr_data,
        n_neighbors=1,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=10,
        verbose=False
    )
    _, dists1 = nnd.query(prev_data,k=1)
    m1 = dists1.mean()
    return m1

class Segmenter:
    def __init__(self, data_provider, threshold, range_s=None, range_e=None, range_p=None):
        self.data_provider = data_provider
        self.threshold = threshold
        if range_s is None:
            self.s = data_provider.s
            self.e = data_provider.e
            self.p = data_provider.p
        else:
            self.s = range_s
            self.e = range_e
            self.p = range_p

    def cal_interval_dists(self):
        interval_num = (self.e - self.s)// self.p

        dists = np.zeros(interval_num)
        for curr_epoch in range(self.s, self.e, self.p):
            next_data = self.data_provider.test_representation(curr_epoch+ self.p)
            curr_data = self.data_provider.test_representation(curr_epoch)
            dists[(curr_epoch-self.s)//self.p] = hausdorff_d(curr_data=next_data, prev_data=curr_data)
        
        self.dists = np.copy(dists)
        return dists
    def segment(self):
        dists_segs = list()
        
        count = 0
        base = len(self.dists)-1
        for i in range(len(self.dists)-1, -1, -1):
            count = count + self.dists[i]
            if count >self.threshold:
                dists_segs.insert(0, (i+1, base))
                base = i
                count = 0
        segs = [(self.s+i*self.p, self.s+(j+1)*self.p) for i, j in dists_segs]
        return segs


        
        


