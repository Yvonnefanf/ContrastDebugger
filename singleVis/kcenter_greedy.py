# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center. namely hausdorff distance

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
from sklearn.metrics import pairwise_distances

class kCenterGreedy(object):

  def __init__(self, X, metric='euclidean'):
    self.features = X
    self.name = 'kcenter'
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.features.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers is not None:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_with_budgets(self, already_selected, budgets):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      budgets: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    print('Calculating distances...')
    t0 = time.time()
    self.update_distances(already_selected, only_new=False, reset_dist=True)
    t1 = time.time()
    print("calculating distances for {:d} points within {:.2f} seconds...".format(len(already_selected), t1 - t0))

    new_batch = []

    for _ in range(budgets):
      ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Hausdorff distance is {:.2f} with {:d} points'.format(self.min_distances.max(), len(already_selected)+len(new_batch)))
    
    self.already_selected = np.concatenate((already_selected, np.array(new_batch)))

    return new_batch

  def select_batch_with_distance(self, already_selected, dist):
      """
      Diversity promoting active learning method that greedily forms a batch
      to minimize the maximum distance to a cluster center among all unlabeled
      datapoints.

      Args:
        budgets: batch size

      Returns:
        indices of points selected to minimize distance to cluster centers
      """

      print('Calculating distances...')
      t0 = time.time()
      self.update_distances(already_selected, only_new=False, reset_dist=True)
      t1 = time.time()
      print("calculating distances for {:d} points within {:.2f} seconds...".format(len(already_selected), t1 - t0))

      new_batch = []

      while True:
        ind = np.argmax(self.min_distances)
        curr_min = self.min_distances[ind]
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
        assert ind not in already_selected

        self.update_distances([ind], only_new=True, reset_dist=False)
        new_batch.append(ind)
        if curr_min<dist:
            break
      print('Hausdorff distance is {:.2f} with {:d} points'.format(self.min_distances.max(), len(already_selected)+len(new_batch)))
      
      # self.already_selected = already_selected.extend(new_batch)
      self.already_selected = np.concatenate((already_selected, np.array(new_batch)))

      return new_batch
