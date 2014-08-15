__kernel void vec_add_gpu(__global const float *src_a, 
                          __global const float *src_b, 
                          __global float *res,
                          const int num) {
    int id = get_global_id(0);
    if (id < num)
        res[id] = src_a[id] + src_b[id];
}
