import numpy as np
#========== Mock data ==============

 # Graph 1: Linear chain (5 nodes) - 0-1-2-3-4
adj1 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 1],
      [0, 0, 0, 1, 0]
  ]
feats1 = [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [4, 4, 4],
      [5, 5, 5]
  ]

# Graph 2: Star with 4 leaves (5 nodes) - 0 center, 1,2,3,4 leaves
adj2 = [
      [0, 1, 1, 1, 1],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0]
  ]
feats2 = [
      [0, 5, 0],  # Center node
      [1, 1, 4],  # Leaf 1
      [2, 2, 3],  # Leaf 2
      [3, 3, 2],  # Leaf 3
      [4, 4, 1]   # Leaf 4
  ]

# Graph 3: Binary tree (5 nodes) - 0 root, 1,2 children of 0, 3,4 children of 1
adj3 = [
      [0, 1, 1, 0, 0],
      [1, 0, 0, 1, 1],
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0]
  ]
feats3 = [
      [2, 3, 2],  # Root
      [1, 2, 3],  # Left child of root
      [3, 4, 1],  # Right child of root
      [0, 1, 4],  # Left child of node 1
      [2, 1, 5]   # Right child of node 1
  ]

# Graph 4: Path with branch (5 nodes) - 0-1-2-3 with 4 connected to 2
adj4 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 1],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0]
  ]
feats4 = [
      [1, 0, 2],
      [2, 1, 3],
      [3, 2, 4],  # Branch point
      [4, 3, 5],
      [3, 4, 3]   # Branch leaf
  ]

# Graph 5: Y-shaped tree (5 nodes) - 0-1-2 spine, with 3,4 connected to 2
adj5 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 1],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0]
  ]
feats5 = [
      [0, 2, 1],
      [1, 3, 2],
      [2, 4, 3],  # Junction point
      [3, 5, 2],  # Branch 1
      [1, 6, 4]   # Branch 2
  ]


cur_conn = [np.array(adj1), np.array(adj2), np.array(adj3), np.array(adj4), np.array(adj5)] 
cur_attr = [np.array(feats1), np.array(feats2), np.array(feats3), np.array(feats4), np.array(feats5)]
