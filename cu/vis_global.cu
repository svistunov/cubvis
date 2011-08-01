#include "cu/common.cu"

__device__ int intersect(float *V, float3 point, float3 dir) {
    for (int i = 0; i < $vN; i +=9) {
        float3 v0 = make_vert(V, i);
        float3 edge1 = make_vert(V, i+3) - v0;
        float3 edge2 = make_vert(V, i+6) - v0;
        if (intr_tringle(v0, edge1, edge2, point, dir) == 0) {
            return 0;
        }
    }
    return 1;
}

__global__ void vis(float *Vis, float *V, float *N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ind = tid*3;
    while (tid < $visN) {
        float sum = 0.0f;
        float3 normal = make_vert(N, ind);
        for (int i = kernelN; i < (kernelN + $kernelStep); i+=4) {
            float3 dir = make_vert(design, i);
            if (dot(normal, dir) > 0) {
                float3 point = make_shift_point(V, ind, normal, 1.1f);
                sum += design[i+3]*intersect(V,point,dir);
            }
        }
        Vis[tid] += 2*sum;//TODO: correct 2* -- wrong
        tid += blockDim.x * gridDim.x;
    }
}
