/*
 * Name:ZENG Yang
 * Student id:20711899
 * ITSC email:yzengav@connect.ust.hk
 *
 * Please only change this file and do not change any other files.
 * Feel free to change/add any helper functions.
 *
 * COMPILE: g++ -lstdc++ -std=c++11 -lpthread clustering_pthread_skeleton.cpp -main.cpp -o pthread
 * RUN:     ./pthread <path> <epsilon> <mu> <num_threads>
 */

#include <pthread.h>
#include "clustering.h"

struct AllThings{
    int num_threads;
    int my_rank;
    int num_vs;
    int *nbr_offs = nullptr;
    int *nbrs = nullptr;
    int *num_sim_nbrs = nullptr;
    int mu;
    float epsilon;
    bool *pivots = nullptr;
    int **sim_nbrs = nullptr;

    AllThings(int inum_threads, int imy_rank, int inum_vs, int imu, float iepsilon, int *inbr_offs = nullptr, int *inbrs = nullptr, 
    int *inum_sim_nbrs = nullptr, bool *ipivots = nullptr, int **isim_nbrs = nullptr){
        num_threads = inum_threads;
        my_rank = imy_rank;
        num_vs = inum_vs;
        nbr_offs = inbr_offs;
        nbrs = inbrs;
        num_sim_nbrs = inum_sim_nbrs;
        mu = imu;
        epsilon = iepsilon;
        pivots = ipivots;
        sim_nbrs = isim_nbrs;
    };
};


void *parallel(void* allthings){
    AllThings *all = (AllThings *) allthings;

    //parallel stage1 
    int num_vs = all->num_vs;
    int *nbr_offs = all->nbr_offs;
    int *nbrs = all->nbrs;
    long myrank = all->my_rank;
    long nthreads = all->num_threads;
    int **sim_nbrs = all->sim_nbrs;
    float epsilon = all->epsilon;
    int *num_sim_nbrs = all->num_sim_nbrs;
    int mu = all->mu;
    bool *pivots = all->pivots;

    //for distribute evenly
    int my_n = num_vs/nthreads;
    int my_fv = myrank*my_n;
    if (myrank == nthreads - 1) my_n = my_n + num_vs%nthreads;
    int my_lv = my_fv + my_n - 1;

    for (int i = my_fv; i <= my_lv; i++) {
    int *left_start = &nbrs[nbr_offs[i]];
    int *left_end = &nbrs[nbr_offs[i + 1]];
    int left_size = left_end - left_start;

    sim_nbrs[i] = new int[left_size];
    // loop over all neighbors of i
    for (int *j = left_start; j < left_end; j++) {
      int nbr_id = *j;

      int *right_start = &nbrs[nbr_offs[nbr_id]];
      int *right_end = &nbrs[nbr_offs[nbr_id + 1]];
      int right_size = right_end - right_start;

      // compute the similarity
      int num_com_nbrs = get_num_com_nbrs(left_start, left_end, right_start, right_end);

      float sim = (num_com_nbrs + 2) / std::sqrt((left_size + 1.0) * (right_size + 1.0));

      if (sim > epsilon) {
        sim_nbrs[i][num_sim_nbrs[i]] = nbr_id;
        num_sim_nbrs[i]++;
      }
    }
    if (num_sim_nbrs[i] > mu) pivots[i] = true;
  }

    return 0;
}

void expansion(int cur_id, int num_clusters, int *num_sim_nbrs, int **sim_nbrs,
               bool *visited, bool *pivots, int *cluster_result) {
  for (int i = 0; i < num_sim_nbrs[cur_id]; i++) {
    int nbr_id = sim_nbrs[cur_id][i];
    if ((pivots[nbr_id])&&(!visited[nbr_id])){
      visited[nbr_id] = true;
      cluster_result[nbr_id] = num_clusters;
      expansion(nbr_id, num_clusters, num_sim_nbrs, sim_nbrs, visited, pivots,
                cluster_result);
    }
  }
}

int *scan(float epsilon, int mu, int num_threads, int num_vs, int num_es, int *nbr_offs, int *nbrs){
    long thread;
    pthread_t* thread_handles = (pthread_t*) malloc(num_threads*sizeof(pthread_t));
    int *cluster_result = new int[num_vs];

    //global vars
    bool *pivots = new bool[num_vs]();
    int *num_sim_nbrs = new int[num_vs]();
    int **sim_nbrs = new int*[num_vs];

    for (thread=0; thread < num_threads; thread++)
        pthread_create(&thread_handles[thread], NULL, parallel, (void *) new AllThings(
            num_threads, thread, num_vs, mu, epsilon, nbr_offs, nbrs, num_sim_nbrs, pivots, sim_nbrs));
    for (thread=0; thread < num_threads; thread++)
        pthread_join(thread_handles[thread], NULL);
    

    bool *visited = new bool[num_vs]();
    std::fill(cluster_result, cluster_result + num_vs, -1);
    for (int i = 0; i < num_vs; i++) {
        if (!pivots[i] || visited[i]) continue;

        visited[i] = true;
        cluster_result[i] = i;
        expansion(i, i, num_sim_nbrs, sim_nbrs, visited, pivots, cluster_result);
    }

    return cluster_result;
}



