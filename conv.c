#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#ifdef HPC
#include <omp.h>
#endif

typedef struct {
  float **data;
  int64_t H, W;
} Matrix;

void free_matrix(Matrix mat) {
  for (int64_t i = 0; i < mat.H; ++i)
    free(mat.data[i]);
  free(mat.data);
}

Matrix create_matrix(int64_t H, int64_t W) {
  Matrix mat = {0};
  mat.H = H;
  mat.W = W;
  mat.data = (float **)malloc(sizeof(float *) * H);
  for (int64_t i = 0; i < H; ++i) {
    mat.data[i] = (float *)malloc(sizeof(float) * W);
    for (int64_t j = 0; j < W; ++j) {
      mat.data[i][j] = rand() / (float)(RAND_MAX + 1.0);
    }
  }
  return mat;
}

void write_matrix(char *path, Matrix mat) {
  FILE* o_file = fopen(path, "w+");
  fprintf(o_file, "%lld %lld\n", mat.H, mat.W);
  for (int64_t i = 0; i < mat.H; ++i)
  {
    for (int64_t j = 0; j < mat.W; ++j) {
      fprintf(o_file, "%.3f ", mat.data[i][j]);
    }
    fprintf(o_file, "\n");
  }
  fclose(o_file);
}

Matrix read_matrix(char *path) {
  Matrix mat = {0};
  FILE *fp = fopen(path, "r");
  fscanf(fp, "%lld %lld", &mat.H, &mat.W);
  mat.data = (float**)malloc(mat.H * sizeof(float*));
  for (int64_t i = 0; i < mat.H; i++) {
    mat.data[i] = (float *)malloc(mat.W * sizeof(float));
    for (int64_t j = 0; j < mat.W; j++) {
      fscanf(fp, "%f", &mat.data[i][j]);
    }
  }
  fclose(fp);
  return mat;
}

void conv2d(
  float **f, // input feature map
  int64_t H, // input height,
  int64_t W, // input width
  float **g, // input kernel
  int64_t kH, // kernel height
  int64_t kW, // kernel width
  float **output
) {
  int64_t diff_H = (int64_t)(kH / 2);
  int64_t diff_W = (int64_t)(kW / 2);

#ifdef HPC
  double tik = omp_get_wtime();
#pragma omp parallel for schedule(guided, 1)
#endif
  for (int64_t i = 0; i < H; ++i) {
    for (int64_t j = 0; j < W; ++j) {
      float local_sum = 0.0;

      // 경계 체크를 루프 밖으로 이동
      const int64_t k_start = (i < diff_H) ? -i : -diff_H;
      const int64_t k_end = (i + diff_H >= H) ? H - 1 - i : diff_H;
      const int64_t l_start = (j < diff_W) ? -j : -diff_W;
      const int64_t l_end = (j + diff_W >= W) ? W - 1 - j : diff_W;
      
      // 최적화된 내부 루프
      for (int64_t k = k_start; k <= k_end; ++k) {
          const float *f_row = f[i + k];  // 포인터 캐싱
          const float *g_row = g[k + diff_H];
          
          for (int64_t l = l_start; l <= l_end; ++l) {
              local_sum += f_row[j + l] * g_row[l + diff_W];
          }
      }

      output[i][j] = local_sum;
    }
  }
#ifdef HPC
  double tok = omp_get_wtime();
  printf("%lld,%.6f,%lld,%lld\n", omp_get_max_threads(), tok - tik, H * W, kH * kW);
#endif
}

int64_t main(int64_t argc, char** argv) {
  int64_t opt, opt_idx;
  char *f_path = NULL, *g_path = NULL, *o_path = NULL;
  int64_t H = 0, W = 0, kH = 0 , kW = 0;

  const struct option longopts[] = {
      {"kH", required_argument, NULL, 'y'},
      {"kW", required_argument, NULL, 'x'},
      {0, 0, 0, 0}};

  while (1)
  {
    opt = getopt_long_only(argc, argv, ":H:W:f:g:o:", longopts, &opt_idx);
    if(opt == -1)
      break;
    switch (opt)
    {
    case 'H':
      H = atoll(optarg);
      break;
    case 'W':
      W = atoll(optarg);
      break;
    case 'y':
      kH = atoll(optarg);
      break;
    case 'x':
      kW = atoll(optarg);
      break;
    case 'f':
      f_path = strdup(optarg);
      break;
    case 'g':
      g_path = strdup(optarg);
      break;
    case 'o':
      o_path = strdup(optarg);
      break;
    default:
      break;
    }
  }

  Matrix f, g;
  FILE *o_file = NULL;
  
  if(H && W && kH && kW) {
    f = create_matrix(H, W);
    g = create_matrix(kH, kW);
  }
  else
  {
    if(f_path)
      f = read_matrix(f_path);
    if(g_path)
      g = read_matrix(g_path);
  }

  Matrix o = create_matrix(f.H, f.W);

  conv2d(f.data, f.H, f.W, g.data, g.H, g.W, o.data);
 
  if(f_path && H && W && g_path && kH && kW) {
    write_matrix(f_path, f);
    write_matrix(g_path, g);
  }

  if(o_path)
    write_matrix(o_path, o);

  if(f_path)
    free(f_path);
  if(g_path)
    free(g_path);
  if(o_path)
    free(o_path);

  free_matrix(f);
  free_matrix(g);
  free_matrix(o);

  return 0;
}