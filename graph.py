




class Edge:
    __id_start = 0
    __id_end = 0
    __val = 0

class Node:
    def __init__(self,id):
        self.__node_id = id
        pass
    __node_id = 0
    __edges = []
    __source = []
    __sink  = []




class Graph:

    __nodes = []

    def __init__(self):
        pass

    def add_node(self):

        self.__nodes.append(len(self.__nodes))
