/* 
 *ZENG Yang
 *20711899
 *yzengav@connect.ust.hk

 * COMPILE: nvcc -std=c++11 clustering_cuda_skeleton.cu clustering_impl.cpp main.cpp -o cuda
 * RUN:     ./cuda <path> <epsilon> <mu> <num_blocks_per_grid> <num_threads_per_block>
 */

#include <iostream>
#include "clustering.h"

// Define variables or functions here
__global__ void kernel(int *d_nbrs, int *d_nbr_offs, bool *d_pivots, int *d_num_sim_nbrs, int *d_sim_nbrs,
    int num_vs, float epsilon, int mu) {
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;
    const int nthread = blockDim.x*gridDim.x;
    //printf("nthread is %d\n", nthread);
    
    //stage 1
    for(int i = tid; i < num_vs; i += nthread) {
        //printf("this is the %dth num_vs\n", i);
        int left_start = d_nbr_offs[i];
        //printf("d_nbr_offs is  %d\n", d_nbr_offs[i]);
        int left_end = d_nbr_offs[i + 1];
        int left_size = left_end - left_start;
        //printf("left_size is %d\n", left_size);

        int cur_pos = d_nbr_offs[i];
        // loop over all neighbors of i
        for (int j = left_start; j < left_end; j++) {
            int nbr_id = d_nbrs[j];

            int right_start = d_nbr_offs[nbr_id];
            int right_end = d_nbr_offs[nbr_id + 1];
            int right_size = right_end - right_start;

            // compute the similarity
            int left_pos = left_start, right_pos = right_start, num_com_nbrs = 0;
        
            while (left_pos < left_end && right_pos < right_end) {
                if (d_nbrs[left_pos] == d_nbrs[right_pos]) {
                    num_com_nbrs++;
                    left_pos++;
                    right_pos++;
                } else if (d_nbrs[left_pos] < d_nbrs[right_pos]) {
                    left_pos++;
                } else {
                    right_pos++;
                }
            }
            
            float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

            if (sim > epsilon) {
                d_sim_nbrs[cur_pos + d_num_sim_nbrs[i]] = nbr_id;
                d_num_sim_nbrs[i]++;
            }
        }
        //printf("compute sim done!\n");
        if (d_num_sim_nbrs[i] > mu){
            d_pivots[i] = true;
        }
    }
/*
    for (int i = 0; i < num_vs; i++){
        printf("pivot is %d", d_pivots[i]);
    }
*/
}

void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int *sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result, int *nbr_offs) {
  for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
    int nbr_id = sim_nbrs[nbr_offs[cur_id] + i];
    if ((pivots[nbr_id])&&(!visited[nbr_id])){
      visited[nbr_id] = true;
      cluster_result[nbr_id] = num_clusters;
      expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                cluster_result, nbr_offs);
    }
  }
}

void cuda_scan(int num_vs, int num_es, int *nbr_offs, int *nbrs,
        float epsilon, int mu, int num_blocks_per_grid, int num_threads_per_block,
        int &num_clusters, int *cluster_result) {

    // Fill in the cuda_scan function here
    //printf("num_vs is %d\n", num_vs);

    bool *h_pivots;
    int *h_num_sim_nbrs;
    int *h_sim_nbrs;
    
    int *d_nbrs;
    int *d_nbr_offs;
    bool *d_pivots;
    int *d_num_sim_nbrs;
    int *d_sim_nbrs;
    
    size_t numvs_bool = num_vs * sizeof(bool);
    size_t numvs_int = num_vs * sizeof(int);
    //size_t size_sim_nbrs = num_vs * num_vs * sizeof(int);
    size_t nbrs_len = (num_es + 1) * sizeof(int);
    //size_t nbrs_len = sizeof(nbrs) / sizeof(int);
    //for (int i = 0; i < num_vs + 1; i++){
    //    printf("nbrs_offs is %d\n", nbr_offs[i]);
    //}
    //printf("nbrs_len is %d\n", nbrs_len);
    size_t nbr_offs_len = (num_vs + 1) * sizeof(int);
    //printf("nbrs_offs_len is %d\n", sizeof (nbr_offs));


    h_pivots = (bool *) malloc(numvs_bool);
    h_num_sim_nbrs = (int *) malloc(numvs_int);
    h_sim_nbrs = (int *) malloc(nbrs_len);

    memset(h_pivots, 0, numvs_bool);
    memset(h_num_sim_nbrs, 0, numvs_int);
    memset(h_sim_nbrs, 0, nbrs_len);
    //for (int i = 0; i < num_vs; i++){
    //    printf("h_num_sim_nbrs is %d\n", h_num_sim_nbrs[i]);
    //}
    
    cudaMalloc(&d_nbrs, nbrs_len);
    cudaMalloc(&d_nbr_offs, nbr_offs_len);
    cudaMalloc(&d_pivots, numvs_bool);
    cudaMalloc(&d_num_sim_nbrs, numvs_int);
    cudaMalloc(&d_sim_nbrs, nbrs_len);

    cudaMemcpy(d_nbrs, nbrs, nbrs_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbr_offs, nbr_offs, nbr_offs_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pivots, h_pivots, numvs_bool, cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_sim_nbrs, h_num_sim_nbrs, numvs_int, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sim_nbrs, h_sim_nbrs, nbrs_len, cudaMemcpyHostToDevice);

    kernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_nbrs, d_nbr_offs, d_pivots, 
        d_num_sim_nbrs, d_sim_nbrs, num_vs, epsilon, mu);

    cudaMemcpy(h_num_sim_nbrs, d_num_sim_nbrs, numvs_int, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sim_nbrs, d_sim_nbrs, nbrs_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pivots, d_pivots, numvs_bool, cudaMemcpyDeviceToHost);
    
    //stage 2
    num_clusters = 0;
    bool *visited = new bool[num_vs]();

    for (int i = 0; i < num_vs; i++) {
        if (!h_pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, h_num_sim_nbrs, h_sim_nbrs, visited, h_pivots, cluster_result, nbr_offs);

        num_clusters ++;
    }
/*
    free(h_pivots);
    free(h_sim_nbrs);
    free(h_num_sim_nbrs);
    free(nbrs);
    free(nbr_offs);
    cudaFree(d_nbrs);
    cudaFree(d_nbr_offs);
    cudaFree(d_pivots);
    cudaFree(d_num_sim_nbrs);
    cudaFree(d_sim_nbrs);
    */
}
