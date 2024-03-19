import numpy as np
import math, random
import sys
from itertools import permutations
import tvm

from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import (
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
)

# m is seq_len, k is hidden_dim
#M, N, K = 1024, 1024, 1024

M_str, N_str, K_str, SM_str = sys.argv[1:5]
M, N, K, SM = int(M_str), int(N_str), int(K_str), int(SM_str)


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [K, N], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float32")

        for i, j, k  in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk].astype(
                        "float32") * B[vk, vj].astype("float32")


def factorize_into_four_numbers(number):
    result = []
    for i in range(1, math.ceil(number**(1/4)) + 1):
        if number % i == 0:
            remaining1 = number // i
            for j in range(i, math.ceil(remaining1**0.333) + 1):
                if remaining1 % j == 0:
                    remaining2 = remaining1 // j
                    for k in range(j, math.ceil(remaining2**0.5) + 1):
                        if remaining2 % k == 0:
                            l = remaining2 // k
                            result.append((i, j, k, l))

    return result

def factorize_into_three_numbers(number):
    result = []

    for i in range(1, math.ceil(number**(1/3) + 1)):
        if number % i == 0:
            remaining = number // i
            for j in range(i, math.ceil(remaining**0.5 + 1)):
                if remaining % j == 0:
                    k = remaining // j
                    result.append((i, j, k))

    return result

def get_ordered_combinations(numbers):
    unique_combinations = set(permutations(numbers, len(numbers)))
    return list(unique_combinations)


def constrained_factorization(sm_target):
    file_name = M_str + "_" + N_str + "_" + K_str + "_" + SM_str + "_tuning_result.txt"
    file_name = "./result_data/" + file_name
    wfile = open(file_name, "w+")
    wfile.write("i0 i1 i2 i3 i4 j0 j1 j2 j3 j4 k0 k1 k2 tflops \n")
    grid_dim_list = factorize_into_four_numbers(sm_target)

    max_tflops = 0
    best_config_str = ""
    for grid_ele in grid_dim_list:
        grid_comb = get_ordered_combinations(grid_ele)
        for grid in grid_comb:
            i_factors = [0] * 5
            j_factors = [0] * 5
            k_factors = [0] * 3

            i0, i1, j0, j1 = grid[0], grid[1], grid[2], grid[3]

            block_x = i0 * i1
            block_y = j0 * j1

            task_num_x = int(M / 16) 
            task_num_y = int(N / 16)
            task_num_k = int(K / 16) 

            if (task_num_x % block_x == 0) and (task_num_y % block_y == 0):
                task_num_x = int(task_num_x / block_x)
                task_num_y = int(task_num_y / block_y)
                i_factors[0] = i0
                i_factors[1] = i1
                j_factors[0] = j0
                j_factors[1] = j1
            else: 
                continue
            ifactors_list_comb = []
            ifactors_list = factorize_into_three_numbers(task_num_x)
            for ifactors in ifactors_list:
                ifactors_list_comb.append(get_ordered_combinations(ifactors))
            
            jfactors_list_comb = []
            jfactors_list = factorize_into_three_numbers(task_num_y)
            for jfactors in jfactors_list:
                jfactors_list_comb.append(get_ordered_combinations(jfactors))

            kfactors_list_comb = []
            kfactors_list = factorize_into_three_numbers(task_num_k)
            for kfactors in kfactors_list:
                kfactors_list_comb.append(get_ordered_combinations(kfactors))
            
            shared_memory_need_x = 0
            shared_memory_need_y = 0

            register_need_x = 0
            register_need_y = 0

            ifactors_cnt = 1
            jfactors_cnt = 1
            kfactors_cnt = 1
            for ifactors in ifactors_list_comb:
                ifactors_cnt *= len(ifactors)
            for jfactors in jfactors_list_comb:
                jfactors_cnt *= len(jfactors)
            for kfactors in kfactors_list_comb:
                kfactors_cnt *= len(kfactors)
            # 只对小的M,N有效, TB 数量少，TB tiling 大，shared memory 需求也就变大
            for ifactors in ifactors_list_comb:
                for ifactor in ifactors:
                    for jfactors in jfactors_list_comb:
                        for jfactor in jfactors:
                            for kfactors in kfactors_list_comb:
                                for kfactor in kfactors:
                                    register_need_x = ifactor[1] * ifactor[2] * kfactor[2] * 16 * 16
                                    register_need_y = jfactor[1] * jfactor[2] * kfactor[2] * 16 * 16
                                    register_need_c = ifactor[1] * jfactor[1] * ifactor[2] * jfactor[2] * 16 * 16
                                    thread_num_y = ifactor[0] * jfactor[0]
                                    # Constraint 1
                                    if thread_num_y * 32 > 1024:
                                        continue
                                    # Constraint 2
                                    per_thread_register = (register_need_c + register_need_x + register_need_y) / thread_num_y / 32

                                    if ((register_need_c + register_need_x + register_need_y) * thread_num_y) > 65536 or per_thread_register > 255 :
                                        continue
                                    shared_memory_need_x = ifactor[0] * register_need_x * kfactor[1]
                                    shared_memory_need_y = jfactor[0] * register_need_y * kfactor[1]
                                    # Constraint 3, get from device_query
                                    if ((shared_memory_need_x + shared_memory_need_y) * 2) > 49152:
                                        continue
                                    i_factors[2] = ifactor[0]
                                    i_factors[3] = ifactor[1]
                                    i_factors[4] = ifactor[2]

                                    j_factors[2] = jfactor[0]
                                    j_factors[3] = jfactor[1]
                                    j_factors[4] = jfactor[2]

                                    k_factors[0] = kfactor[0]
                                    k_factors[1] = kfactor[1]
                                    k_factors[2] = kfactor[2]
                                    
                                    run_config = str(i_factors[0]) + " " + str(i_factors[1]) + " " + \
                                             str(i_factors[2]) + " " + str(i_factors[3]) + " " + \
                                             str(i_factors[4]) + " " + str(j_factors[0]) + " " + \
                                             str(j_factors[1]) + " " + str(j_factors[2]) + " " + \
                                             str(j_factors[3]) + " " + str(j_factors[4]) + " " + \
                                             str(k_factors[0]) + " " + str(k_factors[1]) + " " + \
                                             str(k_factors[2]) + " "                                 
                                    print(run_config + "\n")
                                    
                                    try:
                                        #tflops = run(i_factors, j_factors, k_factors)
                                        #run_config += str(tflops) + " \n"
                                        wfile.write(run_config + "\n")
                                        wfile.flush()
                                        #max_tflops = max(max_tflops, tflops)
                                        #best_config_str = run_config
                                    except Exception as e:
                                        print("run sampling error : ", e)
                                    

    # for SM=64, mnk=1024 best config is i_factos=[4, 2, 2, 4, 1] j_factors=[4, 2, 1, 4, 2] j_factors=[64,1,1]
    print("M = %s, N = %s, K= %s, SM= %s " % (M_str, N_str, K_str, SM_str) )
    #print(best_config_str)
    wfile.close()                



def stochastic_wmma_schedule(
    workload,
    i_factors,
    j_factors,
    k_factors,
    k_inner,
    in_dtype,
    b_transposed,
    ldmatrix_a_intrin,
    ldmatrix_b_intrin,
    wmma_intrin,
    wmma_fill_intrin,
    wmma_store_intrin,
    shared_scope="shared",
):
    """Create a tensorized schedule for GEMM with MMA intrinsics."""
    import tvm  # pylint: disable=import-outside-toplevel

    #ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(workload)

    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, k_inner])

    sch.reorder(i, j, k, i_tc, j_tc, k_tc)

    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block

    #i_factors = sch.sample_perfect_tile(loop=i, n=5)
    #j_factors = sch.sample_perfect_tile(loop=j, n=5)
    #k_factors = sch.sample_perfect_tile(loop=k, n=3)

    #i_factors, j_factors, k_factors = [16, 2, 2, 1, 1], [8, 2, 2, 2, 1], [32, 1, 2]
    
    #i_factors, j_factors, k_factors = [1, 4, 4, 1, 4], [1, 8, 2, 2, 2], [32, 2, 1]  # (1)
    #i_factors, j_factors, k_factors = [1, 4, 4, 2, 2], [16, 1, 1, 4, 1], [16, 1, 4] # （2）
    #i_factors, j_factors, k_factors = [64, 1, 1, 1, 1], [1, 1, 1, 64, 1], [64, 1, 1]

    num_ty = i_factors[2] * j_factors[2]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)

    sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3, k2, i4, j4)

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    warp_size = 32

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, shared_scope)
        sch.compute_at(block_read, k0)
        vector_size = 8
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        offset = 8 if in_dtype == "float16" else 16
        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=offset)

        return block_read

    fetch_to_shared(block_outer, 0, 2)
    fetch_to_shared(block_outer, 1, 2)

    A_warp = sch.cache_read(block_outer, 0, "wmma.matrix_a")
    B_warp = sch.cache_read(block_outer, 1, "wmma.matrix_b")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    C_warp = sch.cache_write(block_outer, 0, "wmma.accumulator")
    sch.reverse_compute_at(C_warp, thread_idy)

    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)

    sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])
    block_init_c = sch.get_block("C_init")

    def tile_wmma_fragment(block_read, height, width):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, width])
        sch.reorder(i0, j0, i1, j1)
        return i1

    loop_a = tile_wmma_fragment(A_warp, 16, k_inner)

    if b_transposed:
        loop_b = tile_wmma_fragment(B_warp, 16, k_inner)
    else:
        loop_b = tile_wmma_fragment(B_warp, k_inner, 16)


    sch.tensorize(loop_a, ldmatrix_a_intrin)
    sch.tensorize(loop_b, ldmatrix_b_intrin)
    sch.tensorize(sch.get_loops(block_inner)[-3], wmma_intrin)
    sch.tensorize(sch.get_loops(block_init_c)[-2], wmma_fill_intrin)
    sch.tensorize(sch.get_loops(C_warp)[-2], wmma_store_intrin)
    #sch.mod.show()

    return sch


def run(i_factors, j_factors, k_factors):
    in_dtype = "float16"
    out_dtype = "float32"
    b_transposed = False
    k_inner = 16
    
    sch = stochastic_wmma_schedule(
        #te.create_prim_func(matmul(M, N, K, in_dtype, out_dtype, b_transposed)),
        MyModule,
        i_factors,
        j_factors,
        k_factors,
        k_inner,
        in_dtype,
        False,  # b_transposed
        WMMA_LOAD_16x16x16_F16_A_INTRIN,
        WMMA_LOAD_16x16x16_F16_B_INTRIN,
        WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
        WMMA_FILL_16x16x16_F32_INTRIN,
        WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
    )

    try:
        f = tvm.build(sch.mod["main"], target="cuda", name="dense")
        #print(f.imported_modules[0].get_source())
        source_code = f.imported_modules[0].get_source()
    except Exception as e:
        print(">>>>>>>>>>>>>>>> build code error <<<<<<<<<<<<<<<<<<", e)

    dev = tvm.device("cuda", 0)

    if in_dtype == "float16":
        a_np = np.random.uniform(size=(M, K)).astype("float16")

        if b_transposed:
            b_np = np.random.uniform(size=(N, K)).astype("float16")
        else:
            b_np = np.random.uniform(size=(K, N)).astype("float16")   

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype=out_dtype), dev)
    f_timer_after = f.time_evaluator(f.entry_name, dev, number=5)
    try:
        result = f_timer_after(a, b, c).mean
    except Exception as e:
        print(">>>>>>>>>>>>>>>>>>>> run code error <<<<<<<<<<<<<<<<<<<<", e)

    tflops = 2 * M * N * K / result / 1e12
    print("============== TFLOPS: %f ======" % tflops)
    return tflops, source_code

def one_test(i_factors, j_factors, k_factors):
    #i_factors, j_factors, k_factors = [4, 4, 2, 2, 1], [2, 2, 8, 2, 1], [24, 2, 1]
    #i_factors, j_factors, k_factors = [1, 4, 4, 2, 2], [16, 1, 1, 4, 1], [16, 1, 4]
    run(i_factors, j_factors, k_factors)


def sampling_from_file(try_number):
    file_name = M_str + "_" + N_str + "_" + K_str + "_" + SM_str + "_tuning_result.txt"
    file_name = "./result_data/" + file_name
    read_file = open(file_name, "r")
    config_list = []
    for line in read_file.readlines():
        config_list.append(line)
    numbers = list(range(1, len(config_list)))
    random.shuffle(numbers)
    sampled_numbers = numbers[:try_number]
    max_tflops = 0
    max_source_code = ""
    best_config = ""
    samping_cnt = 0
    for config_id in sampled_numbers:
        print("====== samping number is: ", samping_cnt)
        samping_cnt += 1
        config = config_list[config_id].replace("\n", "")
        config_items = config.split(" ")
        i_factors = [int(x) for x in config_items[0:5]]
        j_factors = [int(x) for x in config_items[5:10]]
        k_factors = [int(x) for x in config_items[10:13]]
        try:
            tflops, source_code = run(i_factors, j_factors, k_factors)
            print(source_code)
            if tflops > max_tflops:
                max_tflops = tflops
                max_source_code = source_code
                best_config = config
        except Exception as e:
            print("config is:", config)
            print("run code error: ", e)
    config_items = best_config.split(" ")
    i_factors = [int(x) for x in config_items[0:5]]
    j_factors = [int(x) for x in config_items[5:10]]
    k_factors = [int(x) for x in config_items[10:13]]
    one_test(i_factors, j_factors, k_factors)
    print("max tflops is: ", max_tflops)
    print("best config is: ", best_config)
    print("best source code is:", max_source_code)


    

if __name__ == "__main__":
    
    #constrained_factorization(SM)
    #one_test()
    #print(factorize_into_three_numbers(32768))
    sampling_from_file(10)