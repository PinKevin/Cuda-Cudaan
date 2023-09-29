from numba import cuda
import numpy as np
import time


@cuda.jit
def HitungJumlahBagian(start, end, result):
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)

    JumlahBagian = 0
    for i in range(start + thread_id, end + 1, stride):
        if i % 2 == 0:
            JumlahBagian += i

    result[thread_id] = JumlahBagian


if __name__ == "__main__":
    N = 10  # Ganti dengan nilai N yang sesuai

    # Menghitung jumlah thread yang akan digunakan pada GPU
    block_size = 256
    num_threads = 256
    num_blocks = (num_threads + block_size - 1) // block_size

    # Memesan memori untuk hasil per thread di GPU
    d_result = cuda.device_array(num_threads, dtype=np.int32)

    # Menghitung waktu awal eksekusi per thread
    start_time = time.time()

    # Melakukan perhitungan pada GPU
    HitungJumlahBagian[num_blocks, num_threads](1, N, d_result)

    # Mengumpulkan hasil dari GPU
    h_result = np.empty(num_threads, dtype=np.int32)
    d_result.copy_to_host(h_result)

    # Menghitung hasil akhir dari hasil per thread
    total_result = np.sum(h_result)

    # Menghitung waktu akhir eksekusi per thread
    end_time = time.time()
    execution_time = end_time - start_time

    print(
        f'Jumlah semua bilangan genap dari 1 hingga {N} adalah {total_result}. Waktu : {execution_time} detik')
