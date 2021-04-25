//Name: ZENG Yang
//Student ID: 20711899
//Email: yzengav@connect.ust.hk
#include "clustering.h"

#include "mpi.h"

#include <cassert>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  int num_process; // number of processors
  int my_rank;     // my global rank

  comm = MPI_COMM_WORLD;

  MPI_Comm_size(comm, &num_process);
  MPI_Comm_rank(comm, &my_rank);

  if (argc != 3) {
    std::cerr << "usage: ./clustering_sequential data_path result_path"
              << std::endl;

    return -1;
  }
  std::string dir(argv[1]);
  std::string result_path(argv[2]);

  int num_graphs;
  int *clustering_results = nullptr;
  int *num_cluster_total = nullptr;

  int *nbr_offs = nullptr, *nbrs = nullptr;
  int *nbr_offs_local = nullptr, *nbrs_local = nullptr;

  GraphMetaInfo *info = nullptr;

  // read graph info from files
  if (my_rank == 0) {
    num_graphs = read_files(dir, info, nbr_offs, nbrs);
  }
  auto start_clock = chrono::high_resolution_clock::now();

  ////////////////////////////////////////////////////////////////
  // ADD THE CODE HERE
  int* send_offs_counts = new int[num_process];
  int* send_offs_offs = new int[num_process];
  int* send_nbrs_counts = new int[num_process];
  int* send_nbrs_offs = new int[num_process];
  int* recv_res_offs = new int[num_process];
  int* recv_res_counts = new int[num_process];
  //bcast the num_graphs
  MPI_Bcast(&num_graphs,1,MPI_INT,0,comm);
  int local_n = num_graphs/num_process;
  //printf("process %d has %d graphs\n", my_rank, local_n);

  //Define a new MPI_datatype
  MPI_Datatype graph;
  int lengths[2] = {1,1};
  const MPI_Aint displacements[2] = {0, sizeof(int)};
  MPI_Datatype types[2] = {MPI_INT, MPI_INT};
  MPI_Type_create_struct(2, lengths, displacements, types, &graph);
  MPI_Type_commit(&graph);
  //printf("New MPI datatype has been built!\n");

  //build the structure for local info
  GraphMetaInfo *info_pp = nullptr;
  info_pp = (GraphMetaInfo*)calloc(local_n, sizeof(GraphMetaInfo));
  //printf("process %d build MPI datatype successful!\n", my_rank);

  //scatter 
  if (my_rank == 0){
    int offs_temp = 0;
	  int nbrs_temp = 0;
	  int local_offs_len = 0;
	  int local_nbrs_len = 0;
    for (int i=0; i<num_process; i++){ 
      offs_temp += local_offs_len;
      nbrs_temp += local_nbrs_len;
      local_offs_len = 0;
      local_nbrs_len = 0;
      for(int j=0; j<local_n; j++){
        local_offs_len += info[i*local_n+j].num_vertices + 1;
        local_nbrs_len += info[i*local_n+j].num_edges + 1;	
      }
      send_offs_counts[i] = local_offs_len;
      send_nbrs_counts[i] = local_nbrs_len;
      send_offs_offs[i] = offs_temp;
      send_nbrs_offs[i] = nbrs_temp;
    }
  }
  MPI_Bcast(send_offs_counts, num_process, MPI_INT, 0, comm);
  MPI_Bcast(send_nbrs_counts, num_process, MPI_INT, 0, comm);
  MPI_Bcast(send_offs_offs, num_process, MPI_INT, 0, comm);
  MPI_Bcast(send_nbrs_offs, num_process, MPI_INT, 0, comm);

  nbr_offs_local = new int[send_offs_counts[my_rank]];
  nbrs_local = new int[send_nbrs_counts[my_rank]];

  MPI_Scatter(info, local_n, graph,
              info_pp, local_n, graph, 0, comm);
  //printf("process %d scatter info successful!\n", my_rank);
  //cout<<info_pp->num_edges<<endl;

  MPI_Scatterv(nbr_offs, send_offs_counts, send_offs_offs, MPI_INT,
              nbr_offs_local, send_offs_counts[my_rank], MPI_INT, 0, comm);
  MPI_Scatterv(nbrs, send_nbrs_counts, send_nbrs_offs, MPI_INT,
              nbrs_local, send_nbrs_counts[my_rank], MPI_INT, 0, comm);
  //printf("process %d scatter all successful!\n", my_rank);
  //cout<<nbr_offs_local<<endl;
  //cout<<nbrs_local<<endl;

  //clustering
  int *clustering_results_local[local_n];
  int num_cluster_localtotal[local_n];
  if(my_rank == 0){
	  num_cluster_total = new int[num_graphs];
  }

  //find the results list length for each process
  int local_res_length = send_offs_counts[my_rank] - local_n;
  int* res_1dlist = new int[local_res_length];
  int n = 0;

  for (size_t i = 0; i < local_n; i++) {
    GraphMetaInfo info_local = info_pp[i];
    //printf("process %d info_local:", my_rank);
    //cout<<info_local.num_edges<<endl;
        
    clustering_results_local[i] = 
        (int *)calloc(info_local.num_vertices, sizeof(int));
    //printf("process %d calloc done!\n", my_rank);

    int num_cluster_local = clustering(info_local, nbr_offs_local, nbrs_local,
                                      clustering_results_local[i]);
    //printf("process %d clustering done!\n", my_rank);
    num_cluster_localtotal[i] = num_cluster_local;

    nbr_offs_local += (info_local.num_vertices + 1);
    nbrs_local += (info_local.num_edges + 1);

    //In each process, covert results to 1d list
    for (int j = 0; j < info_pp[i].num_vertices; j++){
      res_1dlist[n++] = clustering_results_local[i][j];
    }
  }

  //Gather
  //Gather the total cluster num in each process
  MPI_Gather(num_cluster_localtotal, local_n, MPI_INT, 
            num_cluster_total, local_n, MPI_INT, 0, comm);
  
  //build the counts and offset of gatherv
  MPI_Allgather(&local_res_length, 1, MPI_INT, 
                recv_res_counts, 1, MPI_INT, comm);
  recv_res_offs[0] = 0;
  for (int i = 1; i < num_process; i++){
    recv_res_offs[i] = recv_res_offs[i-1] + recv_res_counts[i-1];
  }

  //Gather the results lists
  if (my_rank == 0){
    int res_final_length = 0;
    for (int i = 0; i < num_process; i++){
      res_final_length += recv_res_counts[i];
    }
    clustering_results = new int[res_final_length];
  }
  MPI_Gatherv(res_1dlist, recv_res_counts[my_rank], MPI_INT, 
            clustering_results, recv_res_counts, recv_res_offs, MPI_INT, 0, comm);
  //End of code
  //////////////////////////////////////////////////////////////////////////
  
  MPI_Barrier(comm);
  auto end_clock = chrono::high_resolution_clock::now();

  // 1) print results to screen
  if (my_rank == 0) {
    for (size_t i = 0; i < num_graphs; i++) {
      printf("num cluster in graph %d : %d\n", i, num_cluster_total[i]);
    }
    fprintf(stderr, "Elapsed Time: %.9lf ms\n",
            chrono::duration_cast<chrono::nanoseconds>(end_clock - start_clock)
                    .count() /
                pow(10, 6));
  }

  // 2) write results to file
  if (my_rank == 0) {
    int *result_graph = clustering_results;
    for (int i = 0; i < num_graphs; i++) {
      GraphMetaInfo info_local = info[i];
      write_result_to_file(info_local, i, num_cluster_total[i], result_graph,
                           result_path);

      result_graph += info_local.num_vertices;
    }
  }

  MPI_Finalize();

  if (my_rank == 0) {
    free(num_cluster_total);
  }

  return 0;

}
