import torch as t

a=t.tensor([[1,2],[3,4]])
b=t.tensor([[5,6],[7,8]])
print(a+b)
print(a-b)
print(b.mm(a))
print(b.storage())
b.reshape(1,4)
print(b.storage())
x=t.ones(3,4,requires_grad=True)
# data,数据，也就是对应的张量
# grad,梯度，也就是Tensor对应的梯度
print(x.data,x.grad,x.grad_fn)
y=x+2
print(y.grad_fn)    # 加法向后AddBackward0
z=t.mean(y.pow(3))  # 取平均向后MeanBackward0
print(z.grad_fn)
z.backward()
print(z.grad)
print(y.grad)
print(x.grad)