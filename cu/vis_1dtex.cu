#include "cu/common.cu"

__device__ int intersect(float3 point, float3 dir) {
    for (int i = 0; i < $vN; i +=9) {
            float3 v0 = make_tex1dvert(v_tex, i);
            float3 edge1 = make_tex1dvert(v_tex, i+3) - v0;
            float3 edge2 = make_tex1dvert(v_tex, i+6) - v0;
            if (intr_tringle(v0, edge1, edge2, point, dir) == 0) {
                return 0;
            }
    }
    return 1;
}

__global__ void vis(float *Vis) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int ind = tid*3;
    while (tid < $visN) {
        float sum = 0.0f;
        float3 normal = make_tex1dvert(n_tex, ind);
        for (int i = kernelN; i < (kernelN + $kernelStep); i+=4) {
            float3 dir = make_vert(design, i);
            if (dot(normal, dir) > -0.01f) { 
                float3 point = make_tex1dshift_point(v_tex, ind, normal, 1.1f);
                sum += design[i+3]*intersect(point,dir);
            }
        }
        Vis[tid] += 2*sum;//TODO: correct 2* -- wrong
        tid += blockDim.x * gridDim.x;
    }
}
