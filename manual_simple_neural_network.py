import numpy as np
import matplotlib.pyplot as plt
class operation():
    def __init__(self,input_nodes=[]):
        self.input_nodes=input_nodes
        self.output_nodes=[]
        for node in input_nodes:
            node.output_nodes.append(self)
        _default_graph.operations.append(self)
    def compute(self):
        pass
class add(operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    def compute(self,x,y):
        self.inputs=[x,y]
        return x + y
class multiply(operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    def compute(self,x,y):
        self.inputs=[x,y]
        return x * y
class matmul(operation):
    def __init__(self,x,y):
        super().__init__([x,y])
    def compute(self,x,y):
        self.inputs=[x,y]
        return x.dot( y)
class placeholder():
    def __init__(self):
        self.output_nodes=[]
        _default_graph.placeholders.append(self)
class variable():
    def __init__(self,initial_value=None):
        self.value=initial_value
        self.output_nodes=[]
        _default_graph.variables.append(self)
class graph():
    def __init__(self):
        self.operations=[]
        self.placeholders=[]
        self.variables=[]
    def set_as_default(self):
        global _default_graph
        _default_graph=self
#####example of making a graph
g=graph()
g.set_as_default()
A=variable(10)
b=variable(1)
x=placeholder()
y=multiply(A,x)
z=add(y,b)

def traverse_postorder(operations):
    nodes_postorder=[]
    def recurse(node):
        if isinstance(node,operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operations)
    return nodes_postorder
class session():
    def run(self,operation,feed_dict={}):
        nodes_postorder=traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node)==placeholder:
                node.output=feed_dict[node]
            elif type(node)==variable:
                node.output=node.value
            else:
                node.inputs=[input_node.output for input_node in node.input_nodes]
                node.output=node.compute(*node.inputs)
            if type(node.output)==list:
                node.output=np.array(node.output)
        return operation.output
sess=session()
result = sess.run(operation=z, feed_dict={x: 10})


class Sigmoid(operation):
    def __init__(self,z):
        super().__init__([z])
        
    def compute(self,z_val):
            return 1/(1+np.exp(-z_val))
from sklearn.datasets import make_blobs
data=make_blobs(n_samples=50,n_features=2,centers=2,random_state=75)
features=data[0]
labels=data[1]
g=graph()
g.set_as_default()
x=placeholder()
w=variable([1,1])
b=variable(-5)
z=add(matmul(w,x),b)
a=Sigmoid(z)
sess=session()
sess.run(operation=a,feed_dict={x:[8,10]})
sess.run(operation=a,feed_dict={x:[2,-10]})

