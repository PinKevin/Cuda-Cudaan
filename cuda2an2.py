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
    # Menentukan ukuran blok dan jumlah blok
    block_size = 10
    num_blocks = 100  # Ganti dengan jumlah blok yang sesuai

    # Menghitung jumlah thread yang akan digunakan pada GPU
    num_threads = block_size * num_blocks

    # Memesan memori untuk hasil per thread di GPU
    d_result = cuda.device_array(num_threads, dtype=np.int32)

    # Menghasilkan N (bilangan bulat positif) secara acak dari NumPy
    # N = np.random.randint(1, 1000001)
    N = 1000000

    # Menghitung waktu awal eksekusi per thread
    start_time = time.time()

    # Melakukan perhitungan pada GPU
    HitungJumlahBagian[num_blocks, block_size](1, N, d_result)

    # Mengumpulkan hasil dari GPU
    h_result = np.empty(num_threads, dtype=np.int32)
    d_result.copy_to_host(h_result)

    # Menghitung waktu akhir eksekusi per thread
    end_time = time.time()
    execution_time = end_time - start_time

    print(
        f'Jumlah semua bilangan genap dari 1 hingga {N} adalah {np.sum(h_result)}. Waktu : {execution_time} detik')

    # Mencetak hasil per blok serta waktu per blok
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = block_start + block_size
        block_sum = np.sum(h_result[block_start:block_end])
        block_time = execution_time / num_blocks
        print(
            f'Block {i}: Jumlah {block_start}-{block_end}: {block_sum}. Waktu: {block_time} detik')
