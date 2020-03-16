#include "collectives.h"
#include "timer.h"
#include "threadpool.h"
#include <mpi.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#include <condition_variable>
#include "thread_pool.hpp"

std::mutex g_mutex;
void testFunc()
{
   // loop to print character after a random period of time
   for (int i = 1; i < 4; ++i)
   {
       std::this_thread::sleep_for(std::chrono::seconds(1));
       std::lock_guard<std::mutex> lock(g_mutex);
       std::cout << "输出 testFunc() [" << i << "] at thread [ " << std::this_thread::get_id() << "] output" << std::endl;
   }

}
void TestCollectivesCPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Initialize on CPU (no GPU device ID).
    InitCollectives(NO_DEVICE);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* data = new float[size];
        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                data[j] = 1.0f;
            }

            float* output;
            timer.start();
            RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(output[j] != (float) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << data[j] << std::endl;
                    return;
                }
            }
            delete[] output;
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        delete[] data;
    }
}



void forward()
{
    printf("前向计算\n");	
    int i ,j;
    for(i=0;i<10000;i++){
        for(j=0;j<600000;j++){

        }
    }
}

void back()
{
    printf("后向传播\n");	
    int i ,j;
    for(i=0;i<10000;i++){
        for(j=0;j<600000;j++){

        }
    }
}

void layer_back()
{
    printf("后向传播\n");	
    int i ,j;
    for(i=0;i<10000;i++){
        for(j=0;j<60000;j++){

        }
    }
}



// 传统的同步训练方式
void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    // 
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    printf("local_ranks : %d\n",mpi_local_rank);
    InitCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    cudaError_t err;

    timer::Timer timer;
    
     sola::thread_pool thread_pool;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* cpu_data = new float[size];

        float* data;
        err = cudaMalloc(&data, sizeof(float) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = 1.0f;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            float* output;
            timer.start();


	    // GPU计算全部层的所有梯度
	    // 同步等待 同步梯度

	    //printf("启动线程池\n");   
	    //thread_pool.add_task(testFunc);
            forward();
            back();
	        // mythreadpool::threadpool executor{ 50 };
    	    // executor.commit(    RingAllreduce,data, size, &output);
	        RingAllreduce(data, size, &output);
            seconds += timer.seconds();
            printf("一次迭代结束\n");

            err = cudaMemcpy(cpu_data, output, sizeof(float) * size, cudaMemcpyDeviceToHost);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(cpu_data[j] != (float) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
                    return;
                }
            }
            err = cudaFree(output);
            if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
    }
}



// 传统的同步训练方式
void Parallel_Ring(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    // 
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    printf("local_ranks : %d\n",mpi_local_rank);
    InitCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    cudaError_t err;

    timer::Timer timer;
    
    sola::thread_pool thread_pool;
    mythreadpool::threadpool executor{1};
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* cpu_data = new float[size];

        float* data;
        err = cudaMalloc(&data, sizeof(float) * size);
        if(err != cudaSuccess) { throw std::runtime_error("cudaMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = 1.0f;
            }

            err = cudaMemcpy(data, cpu_data, sizeof(float) * size, cudaMemcpyHostToDevice);
            if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }

            float* output;
            timer.start();


	        // GPU计算全部层的所有梯度
	        // 同步等待 同步梯度

	        //printf("启动线程池\n");   
	        //thread_pool.add_task(testFunc);

            forward();
            // 逐层并行同步 假设是一个8层的网络，每一层参数64m 全部参数512m
            for(int layer = 8; layer>=0; layer--){
                layer_back();
    	        executor.commit( RingAllreduce,data, size, &output);
            }

            while(executor.get_rest_tasts()!=0 ||  executor.idlCount() != 1){
                printf("等待第一层参数同步\n");
            }
            // 必须等到第一层同步结束之后才算一次迭代完成
	        // RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            printf("一次迭代结束\n");

            // err = cudaMemcpy(cpu_data, output, sizeof(float) * size, cudaMemcpyDeviceToHost);
            // if(err != cudaSuccess) { throw std::runtime_error("cudaMemcpy failed with an error"); }
            // // Check that we get the expected result.
            // for(size_t j = 0; j < size; j++) {
            //     if(cpu_data[j] != (float) mpi_size) {
            //         std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
            //         return;
            //     }
            // }
            // err = cudaFree(output);
            // if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        }

        

        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds / iters
                << " per iteration)" << std::endl;
        }

        err = cudaFree(data);
        if(err != cudaSuccess) { throw std::runtime_error("cudaFree failed with an error"); }
        delete[] cpu_data;
    }
}



// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }
    std::string input(argv[1]);

    // Buffer sizes used for tests.
    // std::vector<size_t> buffer_sizes = {
    //     0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
    // };

    std::vector<size_t> buffer_sizes = {
        67108864, 536870912
    };


    // Number of iterations to run for each buffer size.
/*    std::vector<size_t> iterations = {
        100000, 100000, 100000, 100000,
        1000, 1000, 1000, 1000,
        100, 50, 10, 1
    };
*/   
    // std::vector<size_t> iterations = {
    //     2, 2, 2, 2,
    //     2, 2, 2, 2,
    //     2, 2, 2, 1
    // };


        std::vector<size_t> buffer_sizes = {
            67108864, 536870912
        };
        std::vector<size_t> iterations = {
        2, 2
        };



    // Test on either CPU and GPU.
    if(input == "cpu") {
        TestCollectivesCPU(buffer_sizes, iterations);
    } else if(input == "gpu") {
        TestCollectivesGPU(buffer_sizes, iterations);
    } else if(input == "
    ") {
        td::vector<size_t> buffer_sizes2 = {
            53687091
        };
        std::vector<size_t> iterations2 = {
            100
        };
        Parallel_Ring(buffer_sizes2, iterations2);
    } else {


        std::cerr << "Unknown device type: " << input << std::endl
                  << "Usage: ./allreduce-test (cpu|gpu)" << std::endl;
        return 1;
    }

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
