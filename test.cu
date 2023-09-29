#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Fungsi CUDA untuk menghitung jumlah bilangan genap dalam rentang tertentu
__global__ void HitungJumlahBagian(int start, int end, int* hasil) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;

    int JumlahBagian = 0;
    for (int i = start + tid; i <= end; i += step) {
        if (i % 2 == 0) {
            JumlahBagian += i;
        }
    }

    atomicAdd(hasil, JumlahBagian);
}

int main() {
    int N;
    srand(static_cast<unsigned int>(time(nullptr)));
    N = rand() % 1000000 + 1;

    int* d_hasil;
    cudaMalloc(&d_hasil, sizeof(int));
    cudaMemset(d_hasil, 0, sizeof(int));

    // Konfigurasi blok dan grid CUDA
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Menghitung waktu awal eksekusi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Bagi perhitungan ke dalam beberapa bagian
    int numParts = 10; // Misalnya, kita bagi menjadi 300 bagian
    int partSize = N / numParts;

    int totalHasil = 0; // Variabel untuk menghitung total hasil

    for (int i = 0; i < numParts; ++i) {
        int startPart = i * partSize + 1;
        int endPart = (i != numParts - 1) ? (i + 1) * partSize : N;

        // Memanggil kernel CUDA untuk menghitung jumlah bagian
        HitungJumlahBagian << <gridSize, blockSize >> > (startPart, endPart, d_hasil);
        cudaDeviceSynchronize(); // Menunggu kernel selesai

        // Mengambil hasil dari perangkat CUDA
        int hasil;
        cudaMemcpy(&hasil, d_hasil, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Bagian " << i + 1 << ": Jumlah bilangan genap dari " << startPart << " hingga " << endPart << " adalah " << hasil << std::endl;

        totalHasil += hasil; // Menambahkan hasil sementara ke total
    }

    // Menghitung waktu akhir eksekusi
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Hasil total penghitungan: " << totalHasil << std::endl;
    std::cout << "Waktu eksekusi total: " << milliseconds / 1000.0 << " detik" << std::endl; // Konversi ke detik

    // Membebaskan memori GPU
    cudaFree(d_hasil);

    return 0;
}
