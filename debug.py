# -*- coding: utf-8 -*-
import numpy
import sys
import time
import math
from pprint import pprint
import pycuda.autoinit
import pycuda.driver as cuda_driver
from pycuda.compiler import SourceModule
import os
from Cheetah.Template import Template

def cuda_call_kernal_step(step):
    global secs
    start.record()
    cuda_driver.memcpy_htod(kernel_n, numpy.array(step).astype(numpy.int32))
    cuda_call.prepared_call(grid_dimensions, vis_gpu, pos_gpu,  normals_gpu)
    end.record()
    end.synchronize()
    cuda_driver.Context.synchronize() # ????/???/
    secs = secs + start.time_till(end)*1e-3


start = cuda_driver.Event()
end = cuda_driver.Event()
secs = 0
kernelStep = 8
design = numpy.loadtxt('data/des3d_240_21.txt').astype(numpy.float32)
verts = numpy.loadtxt("data/verts.dat").astype(numpy.float32)
normals = numpy.loadtxt("data/normals.dat").astype(numpy.float32)
indexes = numpy.loadtxt("data/indexes.dat").astype(numpy.ushort)
vis = numpy.zeros((1, verts.size/3), numpy.float32)
template_params = {'dN' : design.size, 'visN': vis.size, 'vN' : verts.size, 'kernelStep' : kernelStep}
kernel_code = Template(
    file = 'vis.cu', 
    searchList = [template_params],
  )
cuda_module = SourceModule(kernel_code)
cuda_call = cuda_module.get_function("vis")
cuda_call.prepare("PPP", (256, 1, 1))
N =  vis.size
grid_dimensions   =  (min(32, (N+256-1) // 256 ), 1)
vis_gpu = cuda_driver.mem_alloc(vis.nbytes)
pos_gpu = cuda_driver.mem_alloc(verts.nbytes)
design_gpu = cuda_module.get_global('design')[0]
cuda_driver.memcpy_htod(design_gpu, design)
design_gpu = cuda_driver.mem_alloc(design.nbytes)
normals_gpu = cuda_driver.mem_alloc(normals.nbytes)
cuda_driver.memcpy_htod(vis_gpu, vis)
cuda_driver.memcpy_htod(pos_gpu, verts)
cuda_driver.memcpy_htod(normals_gpu, normals)
kernel_n = cuda_module.get_global('kernelN')[0]

secs = 0
for i in range(0, design.size, kernelStep):
    cuda_call_kernal_step(i)


print "cuda time: %fs" % secs



res = numpy.empty_like(vis)
cuda_driver.memcpy_dtoh(res, vis_gpu)
numpy.savetxt("data/vis.dat", res)
