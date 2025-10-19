#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

enum
{
  META_TAG = 0,
  DATA_TAG
};

enum
{
  KERNEL_H = 0,
  KERNEL_W,
  STRIDE_H,
  STRIDE_W
};

enum
{
  OPT_kH = 1000,
  OPT_kW,
  OPT_sH,
  OPT_sW
};

typedef struct {
  float **data;
  int64_t H, W;
} Matrix;

void free_matrix(Matrix mat) {
  if(!mat.data)
    return;
  free(mat.data[0]);
  free(mat.data);
}

Matrix create_matrix(int64_t H, int64_t W) {
  Matrix mat = {0};
  mat.H = H;
  mat.W = W;
  
  float *data_block = (float *)malloc(sizeof(float) * H * W);
  mat.data = (float **)malloc(sizeof(float *) * H);
  for (int64_t i = 0; i < H; ++i) {
    mat.data[i] = &(data_block[i * W]);
  }

  for (int64_t i = 0; i < H; ++i) {
    for (int64_t j = 0; j < W; ++j) {
      mat.data[i][j] = rand() / (float)(RAND_MAX + 1.0);
    }
  }
  return mat;
}

void write_matrix(char *path, Matrix mat) {
  FILE* o_file = fopen(path, "w+");
  fprintf(o_file, "%lld %lld\n", mat.H, mat.W);
  for (int64_t i = 0; i < mat.H; ++i) {
    for (int64_t j = 0; j < mat.W; ++j) {
      fprintf(o_file, "%.3f ", mat.data[i][j]);
    }
    fprintf(o_file, "\n");
  }
  fclose(o_file);
}

Matrix read_matrix(char *path) {
  int64_t H, W;
  FILE *fp = fopen(path, "r");
  fscanf(fp, "%lld %lld", &H, &W);
  Matrix mat = create_matrix(H, W);
  for (int64_t i = 0; i < mat.H; i++) {
    for (int64_t j = 0; j < mat.W; j++) {
      fscanf(fp, "%f", &mat.data[i][j]);
    }
  }
  fclose(fp);
  return mat;
}

// Replace your conv2d function with this new, corrected version
Matrix conv2d(
    Matrix input,
    Matrix kernel,
    int64_t stride_height,
    int64_t stride_width,
    int64_t start_offset,
    int64_t *flops_count
)
{   
  int64_t diff_H = (int64_t)(kernel.H / 2);
  int64_t diff_W = (int64_t)(kernel.W / 2);
  
  int64_t num_strides_H = (input.H - start_offset + stride_height - 1) / stride_height;
  int64_t output_height = num_strides_H;
  int64_t output_width = (input.W + stride_width - 1) / stride_width;

  Matrix output = create_matrix(output_height, output_width);
  
  // Initialize FLOPS counter
  *flops_count = 0;
  int64_t total_flops = 0;
  
  // The main loop now starts from the corrected `first_i`.
#ifdef OMP
  #pragma omp parallel reduction(+:total_flops)
#endif
  for (int64_t i = start_offset; i < input.H; i += stride_height) {

#ifdef OMP
  #pragma omp for schedule(static, 16) nowait
#endif 
    for (int64_t j = 0; j < input.W; j += stride_width) {
      float local_sum = 0.0;
      
      const int64_t k_start = (i < diff_H) ? -i : -diff_H;
      const int64_t k_end = (i + diff_H >= input.H) ? input.H - 1 - i : diff_H;
      const int64_t l_start = (j < diff_W) ? -j : -diff_W;
      const int64_t l_end = (j + diff_W >= input.W) ? input.W - 1 - j : diff_W;
      
      for (int64_t k = k_start; k <= k_end; ++k) {
        const float *f_row = input.data[i + k];
        
        int64_t kernel_row = k + diff_H;
        if (kernel_row < 0 || kernel_row >= kernel.H) {
          continue;
        }
        const float *g_row = kernel.data[kernel_row];
        
        for (int64_t l = l_start; l <= l_end; ++l) {
          int64_t kernel_col = l + diff_W;
          if (kernel_col < 0 || kernel_col >= kernel.W) {
            continue;
          }
          local_sum += f_row[j + l] * g_row[kernel_col];
          total_flops += 2; // 1 multiply + 1 add per iteration
        }
      }
      output.data[(i - start_offset) / stride_height][j / stride_width] = local_sum;
    }
  }
  
  // Set the final FLOPS count
  *flops_count = total_flops;
  return output;
}

int main(int argc, char** argv) {
  int rank, size;
  double tik, tok, total_start, total_end;
  double computation_time;
  int64_t proc_flops = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Start total walltime measurement
  total_start = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  if(rank != 0) { 
    // Receive broadcasted kernel info
    int64_t kernel_meta[4];
    MPI_Bcast(kernel_meta, 4, MPI_INT64_T, 0, MPI_COMM_WORLD);
    int64_t kernel_h = kernel_meta[KERNEL_H];
    int64_t kernel_w = kernel_meta[KERNEL_W];
    int64_t stride_h = kernel_meta[STRIDE_H];
    int64_t stride_w = kernel_meta[STRIDE_W];
    Matrix kernel = create_matrix(kernel_h, kernel_w);
    MPI_Bcast(kernel.data[0], kernel_h * kernel_w, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Receive metadata about the specific chunk for this worker
    int64_t input_meta[4];
    MPI_Recv(input_meta, 4, MPI_INT64_T, 0, META_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int64_t input_h = input_meta[0];
    int64_t input_w = input_meta[1];
    int64_t send_start_row = input_meta[2];
    int64_t expected_output_rows = input_meta[3];
    
    // Receive the padded input data
    Matrix local_input = create_matrix(input_h, input_w);
    MPI_Recv(local_input.data[0], input_h * input_w, MPI_FLOAT, 0, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int64_t border = send_start_row + (int64_t)(kernel.H / 2);
    int64_t next_hop = ((int64_t)(border + stride_h - 1) / stride_h) * stride_h;
    int64_t offset = next_hop - send_start_row;

    tik = MPI_Wtime();

    int64_t local_flops = 0;
    Matrix local_output = conv2d(local_input, kernel, stride_h, stride_w, offset, &local_flops);
    int64_t expected_elements = expected_output_rows * local_output.W;

    tok = MPI_Wtime();
    
    computation_time = tok - tik;
    proc_flops = local_flops;

    MPI_Send(local_output.data[0], expected_elements, MPI_FLOAT, 0, DATA_TAG, MPI_COMM_WORLD);
    free_matrix(local_input);
    free_matrix(kernel);
    free_matrix(local_output);
  } 
  else { 
    int opt, opt_idx;
    char *f_path = NULL, *g_path = NULL, *o_path = NULL;
    int64_t H = 0, W = 0, kH = 0 , kW = 0, sH = 1, sW = 1;
    const struct option longopts[] = {
        {"kH", required_argument, NULL, OPT_kH},
        {"kW", required_argument, NULL, OPT_kW},
        {"sH", required_argument, NULL, OPT_sH},
        {"sW", required_argument, NULL, OPT_sW},
        {0, 0, 0, 0}};
    while ((opt = getopt_long_only(argc, argv, ":H:W:f:g:o:", longopts, &opt_idx)) != -1) {
      switch (opt) {
      case 'H': H = atoll(optarg); break;
      case 'W': W = atoll(optarg); break;
      case OPT_kH: kH = atoll(optarg); break;
      case OPT_kW: kW = atoll(optarg); break;
      case OPT_sH: sH = atoll(optarg); break;
      case OPT_sW: sW = atoll(optarg); break;
      case 'f': f_path = strdup(optarg); break;
      case 'g': g_path = strdup(optarg); break;
      case 'o': o_path = strdup(optarg); break;
      default: break;
      }
    }
    Matrix input, kernel;
    if(H && W && kH && kW) {
      input = create_matrix(H, W);
      kernel = create_matrix(kH, kW);
    } else {
      if(f_path) input = read_matrix(f_path);
      if(g_path) kernel = read_matrix(g_path);
    }
  
    // Broadcast kernel info to all workers
    int64_t kernel_meta[4] = {kernel.H, kernel.W, sH, sW};
    MPI_Bcast(kernel_meta, 4, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(kernel.data[0], kernel.H * kernel.W, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- (Work distribution setup is unchanged) ---
    int64_t diff = kernel.H / 2;
    int64_t output_height = (input.H + sH - 1) / sH;
    int64_t output_width = (input.W + sW - 1) / sW;
    Matrix output = create_matrix(output_height, output_width);
    int64_t rows_per_proc = output.H / size;
    int64_t remainder = output.H % size;

    // --- Distribute work and Gather results ---
    int64_t current_output_row = 0;
    for (int i = 0; i < size; ++i) {
      int64_t output_rows_for_this_proc = rows_per_proc + ((i < remainder) ? 1 : 0);
      if(output_rows_for_this_proc == 0) continue;

      int64_t input_start_row = current_output_row * sH;
      int64_t input_end_row = (current_output_row + output_rows_for_this_proc - 1) * sH;
      int64_t send_start_row = (input_start_row - diff > 0) ? (input_start_row - diff) : 0;
      int64_t send_end_row = (input_end_row + diff) < input.H ? (input_end_row + diff) : input.H - 1;
      int64_t send_row_count = send_end_row - send_start_row + 1;
      int64_t ghost_rows_top = input_start_row - send_start_row;
      
      if(i == 0) { // Root process does its own work
        Matrix local_input = create_matrix(send_row_count, input.W);
        memcpy(local_input.data[0], input.data[send_start_row], send_row_count * input.W * sizeof(float));

        tik = MPI_Wtime();

        int64_t local_flops = 0;
        Matrix local_output = conv2d(local_input, kernel, sH, sW, send_start_row, &local_flops);        
        memcpy(output.data[current_output_row], local_output.data[0], local_output.H * local_output.W * sizeof(float));

        tok = MPI_Wtime();
        computation_time = tok - tik;
        proc_flops += local_flops;

        free_matrix(local_input);
        free_matrix(local_output);
      } else { // Send to worker and receive result
        int64_t input_meta[4] = {send_row_count, input.W, send_start_row, output_rows_for_this_proc};
        MPI_Send(input_meta, 4, MPI_INT64_T, i, META_TAG, MPI_COMM_WORLD);
        MPI_Send(input.data[send_start_row], send_row_count * input.W, MPI_FLOAT, i, DATA_TAG, MPI_COMM_WORLD);

        // Receive the trimmed result directly into the correct place in the final output matrix
        float* receive_buffer = output.data[current_output_row];
        MPI_Recv(receive_buffer, output_rows_for_this_proc * output_width, MPI_FLOAT, i, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);       
      }
      current_output_row += output_rows_for_this_proc;
    }

    // --- (File writing and cleanup is unchanged) ---
    if (f_path && H && W && g_path && kH && kW) {
      write_matrix(f_path, input);
      write_matrix(g_path, kernel);
    }
    if(o_path) write_matrix(o_path, output);
    if(f_path) free(f_path);
    if(g_path) free(g_path);
    if(o_path) free(o_path);
    free_matrix(input);
    free_matrix(kernel);
    free_matrix(output);
  }
  
  // Synchronize all processes before final timing
  MPI_Barrier(MPI_COMM_WORLD);
  total_end = MPI_Wtime();
  
  // Collect FLOPS and timing statistics
  int64_t total_flops;
  double max_computation_time;
  
  MPI_Reduce(&proc_flops, &total_flops, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
  if (rank == 0) {
    printf("%lld %.6f\n", total_flops, max_computation_time);
  }

  MPI_Finalize();
  return 0;
}