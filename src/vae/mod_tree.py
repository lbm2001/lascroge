#========== ModTree and TreeNode ==========

class TreeNode(object):  

    def __init__(self, attr):  

        self.neighbors = []  
        self.features = attr  
        
    def add_neighbor(self, nei_node):  
        self.neighbors.append(nei_node)  

class ModTree(object):  

    def __init__(self, attr, conn):  
        self.nodes = []  
        for i,c in enumerate(attr):  
            node = TreeNode(c)  
            self.nodes.append(node)  

        for i in range(len(attr)):  
            for j in range(i + 1, len(attr)):  
                if conn[i, j] != 0:  
                    self.nodes[i].add_neighbor(self.nodes[j])  
                    self.nodes[j].add_neighbor(self.nodes[i])  


        for i,node in enumerate(self.nodes):  
            node.nid = i + 1  
            node.is_leaf = (len(node.neighbors) == 1)  

    def size(self):  
        return len(self.nodes)  