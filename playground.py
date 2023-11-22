import time

import torch

time_cost = []
for _ in range(100):
    start = time.time()
    x = torch.randn([30000, 1000], requires_grad = True)
    z = torch.randn([30000, 1000], requires_grad = False)
    y = x[:3, :] * z[:3, :]
    # 定义损失函数
    loss = (y ** 2).sum()
    # loss.backward()
    # print("Gradient w.r.t. x:", x.grad[:3, :])

    grads = torch.autograd.grad(loss, [x])
    print("Gradient w.r.t. x:", grads[0][:3, :])
    time_cost.append(time.time() - start)
print("time: ", sum(time_cost)/len(time_cost))
