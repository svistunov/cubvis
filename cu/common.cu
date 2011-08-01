#raw
#define EPSILON 0.000001f
#end raw

texture<float, 1, cudaReadModeElementType> v_tex;

texture<float, 1, cudaReadModeElementType> n_tex;

__device__ __constant__ float design[$dN];

__device__ __constant__ int kernelN;

inline __host__ __device__ float3 operator*(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(float3 a, float3 b)
{ 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); 
}

inline __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 make_vert(float *V, int i) {
    return make_float3(V[i], V[i+1], V[i+2]);
}

inline __device__ float3 make_tex1dvert(texture<float, 1, cudaReadModeElementType> t, int i) {
    return make_float3(tex1Dfetch(t, i), tex1Dfetch(t, i+1), tex1Dfetch(t, i+2));
}

inline __device__ float3 make_shift_point(float *V, int i, float3 normal, float val) {
    return make_float3(V[i]+normal.x*val, V[i+1]+normal.y*val, V[i+2]+normal.z*val);
}

inline __device__ float3 make_tex1dshift_point(texture<float, 1, cudaReadModeElementType> t, int i, float3 normal, float val) {
    return make_float3(tex1Dfetch(t, i)+normal.x*val, tex1Dfetch(t, i+1)+normal.y*val, tex1Dfetch(t, i+2)+normal.z*val);
}

inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}



inline __device__ float3 normalize(float3 v) {
    float invLen = rsqrtf(dot(v, v));
    return make_float3(invLen * v.x, invLen * v.y, invLen * v.z);
}

//http://www.cs.virginia.edu/~gfx/Courses/2003/ImageSynthesis/papers/Acceleration/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
__device__ int intr_tringle(float3 v0, float3 edge1, float3 edge2, float3 orig, float3 dir) {
    float3 tvec = orig - v0;
    float3 pvec = cross(dir, edge2);
    float  det  = dot(edge1, pvec);
    det = __fdividef(1.0f, det);

    float u = dot(tvec, pvec) * det;
    if (u < 0.0f || u > 1.0f)
        return 1;

    float3 qvec = cross(tvec, edge1);
    float v = dot(dir, qvec) * det;
    if (v < 0.0f || (u + v) > 1.0f)
        return 1;
        
    return 0;
}
