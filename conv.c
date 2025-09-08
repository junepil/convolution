#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#ifdef HPC
#include <omp.h>
#endif

typedef struct {
  float **data;
  int H, W;
} Matrix;

void free_matrix(Matrix mat) {
  for (int i = 0; i < mat.H; ++i)
    free(mat.data[i]);
  free(mat.data);
}

Matrix create_matrix(int H, int W) {
  Matrix mat = {0};
  mat.H = H;
  mat.W = W;
  mat.data = (float **)malloc(sizeof(float *) * H);
  for (int i = 0; i < H; ++i) {
    mat.data[i] = (float *)malloc(sizeof(float) * W);
    for (int j = 0; j < W; ++j) {
      mat.data[i][j] = 0;
    }
  }

  return mat;
}

void write_matrix(char *path, Matrix mat) {
  FILE* o_file = fopen(path, "w+");
  fprintf(o_file, "%d %d\n", mat.H, mat.W);
  for (int i = 0; i < mat.H; ++i)
  {
    for (int j = 0; j < mat.W; ++j) {
      fprintf(o_file, "%.3f ", mat.data[i][j]);
    }
    fprintf(o_file, "\n");
  }
  fclose(o_file);
}

Matrix read_matrix(char *path) {
  Matrix mat = {0};
  FILE *fp = fopen(path, "r");
  fscanf(fp, "%d %d", &mat.H, &mat.W);
  mat.data = (float**)malloc(mat.H * sizeof(float*));
  for (int i = 0; i < mat.H; i++) {
    mat.data[i] = (float *)malloc(mat.W * sizeof(float));
    for (int j = 0; j < mat.W; j++) {
      fscanf(fp, "%f", &mat.data[i][j]);
    }
  }
  fclose(fp);
  return mat;
}

void conv2d(
  float **f, // input feature map
  int H, // input height,
  int W, // input width
  float **g, // input kernel
  int kH, // kernel height
  int kW, // kernel width
  float **output
) {

  int diff_H = (int)(kH / 2);
  int diff_W = (int)(kW / 2);
#ifdef HPC  
#pragma omp parallel for
#endif
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      float reduce = 0;
      for (int k = -diff_H; k <= diff_H; ++k) {
        for (int l = -diff_W; l <= diff_W; ++l) {
          float temp;
          int f_y = i + k;
          int f_x = j + l;
          
          //* Implement padding
          if (f_y < 0 || f_y >= H || f_x < 0 || f_x >= W) {
            temp = 0;
          } else {
            temp = f[f_y][f_x];
          }
          reduce += temp * g[diff_H + k][diff_W + l];
        }
      }
      output[i][j] = reduce;
    }
  }
}

int main(int argc, char** argv) {
  int opt, opt_idx;
  char *f_path = NULL, *g_path = NULL, *o_path = NULL;
  int H = 0, W = 0, kH = 0 , kW = 0;

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
      H = atoi(optarg);
      break;
    case 'W':
      W = atoi(optarg);
      break;
    case 'y':
      kH = atoi(optarg);
      break;
    case 'x':
      kW = atoi(optarg);
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

  } else {
    if(f_path)
      f = read_matrix(f_path);
    if(g_path)
      g = read_matrix(g_path);
  }

  Matrix o = {
    .data = NULL,
    .H = f.H,
    .W = f.W
  };

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