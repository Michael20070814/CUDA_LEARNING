import torch

a = torch.tensor([1., 2., 3.])

print(torch.cuda.is_available())

print(torch.square(a))
print(a ** 2)
print(a * a)

def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    # CUDA是异步的，所以你不能使用python的时间模块，而应该使用CUDA Event
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup (防止CUDA Context初始化影响时间记录的准确性)
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    # 程序完成之后需要做一次 CUDA 同步
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

time_pytorch_function(torch.square, b)
time_pytorch_function(square_2, b)
time_pytorch_function(square_3, b)

print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_device=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_device=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))