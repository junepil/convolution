#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <mpi.h>

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
  free(mat.data[0]);
  free(mat.data);
}

Matrix create_matrix(int64_t H, int64_t W) {
  Matrix mat = {0};
  mat.H = H;
  mat.W = W;
  mat.data = (float **)malloc(sizeof(float *) * H);
  mat.data[0] = (float *)malloc(sizeof(float) * H * W);
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

Matrix conv2d(
  Matrix input,
  Matrix kernel,
  int64_t stride_height,
  int64_t stride_width)
{
  int64_t diff_H = (int64_t)(kernel.H / 2);
  int64_t diff_W = (int64_t)(kernel.W / 2);

  int64_t output_height = (input.H + stride_height - 1) / stride_height;
  int64_t output_width = (input.W + stride_width - 1) / stride_width;

  Matrix output = create_matrix(output_height, output_width);

  for (int64_t i = 0; i < input.H; i += stride_height) {
    for (int64_t j = 0; j < input.W; j += stride_width) {
      float local_sum = 0.0;

      // 경계 체크를 루프 밖으로 이동
      const int64_t k_start = (i < diff_H) ? -i : -diff_H;
      const int64_t k_end = (i + diff_H >= input.H) ? input.H - 1 - i : diff_H;
      const int64_t l_start = (j < diff_W) ? -j : -diff_W;
      const int64_t l_end = (j + diff_W >= input.W) ? input.W - 1 - j : diff_W;
      
      // 최적화된 내부 루프
      for (int64_t k = k_start; k <= k_end; ++k) {
          const float *f_row = input.data[i + k];  // 포인터 캐싱
          const float *g_row = kernel.data[k + diff_H];
          
          for (int64_t l = l_start; l <= l_end; ++l) {
              local_sum += f_row[j + l] * g_row[l + diff_W];
          }
      }

      output.data[i / stride_height][j / stride_width] = local_sum;
    }
  }

  return output;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int opt, opt_idx;
  char *f_path = NULL, *g_path = NULL, *o_path = NULL;
  int64_t H = 0, W = 0, kH = 0 , kW = 0, sH = 1, sW = 1;

  const struct option longopts[] = {
      {"kH", required_argument, NULL, OPT_kH},
      {"kW", required_argument, NULL, OPT_kW},
      {"sH", required_argument, NULL, OPT_sH},
      {"sW", required_argument, NULL, OPT_sW},
      {0, 0, 0, 0}};

  while (1)
  {
    opt = getopt_long_only(argc, argv, ":H:W:f:g:o:", longopts, &opt_idx);
    if(opt == -1)
      break;
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
  FILE *o_file = NULL;
  
  if(H && W && kH && kW) {
    input = create_matrix(H, W);
    kernel = create_matrix(kH, kW);
  }
  else
  {
    if(f_path)
      input = read_matrix(f_path);
    if(g_path)
      kernel = read_matrix(g_path);
  }

  Matrix output = conv2d(input, kernel, sH, sW);
 
  if(f_path && H && W && g_path && kH && kW) {
    write_matrix(f_path, input);
    write_matrix(g_path, kernel);
  }

  if(o_path)
    write_matrix(o_path, output);

  if(f_path)
    free(f_path);
  if(g_path)
    free(g_path);
  if(o_path)
    free(o_path);

  free_matrix(input);
  free_matrix(kernel);
  free_matrix(output);
  MPI_Finalize();
  return 0;
}