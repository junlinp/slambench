// Simple loader program for BAL-style Dubrovnik problem, matching msckf.c.

#include "linalg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

template<int residual_dim, int camera_dim, int point_dim>
void shur_construction(double *Hc, double *Hp, double *A) {
    // Hf is residual_x_point_dim matrix,
    // Hc is residual_x_camera_dim matrix
    // A is camera_dim x point_dim matrix
    for (int r = 0; r < camera_dim; ++r) {
        for (int c = 0; c < point_dim; ++c) {
            double s = 0.0;
            for (int k = 0; k < residual_dim; ++k) {
                s += Hp[k * point_dim + c] * Hc[k * camera_dim + r];
            }
            A[r * point_dim + c] = s;
        }
    }
}

void inv_3x3(const double *A, double *A_inv) {
  double det = A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
  A_inv[0] = (A[4] * A[8] - A[5] * A[7]) / det;
  A_inv[1] = (A[2] * A[7] - A[1] * A[8]) / det;
  A_inv[2] = (A[1] * A[5] - A[2] * A[4]) / det;
  A_inv[3] = (A[5] * A[6] - A[3] * A[8]) / det;
  A_inv[4] = (A[0] * A[8] - A[2] * A[6]) / det;
  A_inv[5] = (A[2] * A[3] - A[0] * A[5]) / det;
  A_inv[6] = (A[3] * A[7] - A[4] * A[6]) / det;
  A_inv[7] = (A[0] * A[6] - A[2] * A[4]) / det;
  A_inv[8] = (A[0] * A[4] - A[1] * A[3]) / det;
}

void outer_product(const double *A, int m, int n, double *C) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double s = 0.0;
      for (int k = 0; k < m; ++k) {
        s += A[k * n + i] * A[k * n + j];
      }
      C[i * n + j] = s;
    }
  }
}

typedef struct {
  double aa[3]; // angle-axis
  double t[3];  // translation
  double f;     // focal
  double k1, k2;
} Camera;

typedef struct {
  int cam;
  int point;
  double u;
  double v;
} Observation;

typedef struct {
  double p[3];
} Point;

typedef struct {
  int count;
  int *indices;
} PointObs;

static void skew(const double v[3], double S[9]) {
  S[0] = 0;
  S[1] = -v[2];
  S[2] = v[1];
  S[3] = v[2];
  S[4] = 0;
  S[5] = -v[0];
  S[6] = -v[1];
  S[7] = v[0];
  S[8] = 0;
}

static double wall_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void project(const Camera *cam, const double R[9], const double Pw[3], double *u, double *v) {
  double Xc[3];
  mat3_mul_vec(R, Pw, Xc);
  Xc[0] += cam->t[0];
  Xc[1] += cam->t[1];
  Xc[2] += cam->t[2];
  double invz = 1.0 / Xc[2];
  // Bundler/Ceres convention uses negative z forward
  double xn = -Xc[0] * invz;
  double yn = -Xc[1] * invz;
  double r2 = xn * xn + yn * yn;
  double distortion = 1.0 + cam->k1 * r2 + cam->k2 * r2 * r2;
  *u = cam->f * distortion * xn;
  *v = cam->f * distortion * yn;
}

static void jacobians(const Camera *cam, const double R[9], const double Pw[3], double J_cam[18],
                      double J_point[6]) {
  double Xc[3];
  double RPw[3];
  mat3_mul_vec(R, Pw, RPw);
  Xc[0] = RPw[0];
  Xc[1] = RPw[1];
  Xc[2] = RPw[2];
  Xc[0] += cam->t[0];
  Xc[1] += cam->t[1];
  Xc[2] += cam->t[2];
  double invz = 1.0 / Xc[2];
  double xn = -Xc[0] * invz;
  double yn = -Xc[1] * invz;

  double r2 = xn * xn + yn * yn;
  double g = 1.0 + cam->k1 * r2 + cam->k2 * r2 * r2;
  double dg_dxn = cam->k1 * 2.0 * xn + cam->k2 * 4.0 * r2 * xn;
  double dg_dyn = cam->k1 * 2.0 * yn + cam->k2 * 4.0 * r2 * yn;

  double f = cam->f;

  double dpx_dxn = f * (g + xn * dg_dxn);
  double dpx_dyn = f * (xn * dg_dyn);
  double dpy_dxn = f * (yn * dg_dxn);
  double dpy_dyn = f * (g + yn * dg_dyn);

  double d_xn_dX = -invz;
  double d_xn_dZ = Xc[0] * invz * invz;
  double d_yn_dY = -invz;
  double d_yn_dZ = Xc[1] * invz * invz;

  double Jpi[6];
  Jpi[0] = dpx_dxn * d_xn_dX;
  Jpi[1] = dpx_dyn * d_yn_dY;
  Jpi[2] = dpx_dxn * d_xn_dZ + dpx_dyn * d_yn_dZ;
  Jpi[3] = dpy_dxn * d_xn_dX;
  Jpi[4] = dpy_dyn * d_yn_dY;
  Jpi[5] = dpy_dxn * d_xn_dZ + dpy_dyn * d_yn_dZ;

  double S[9];
  skew(RPw, S);
  double A[18];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      A[r * 6 + c] = -S[r * 3 + c];
    }
  }
  A[0 * 6 + 3] = 1;
  A[0 * 6 + 4] = 0;
  A[0 * 6 + 5] = 0;
  A[1 * 6 + 3] = 0;
  A[1 * 6 + 4] = 1;
  A[1 * 6 + 5] = 0;
  A[2 * 6 + 3] = 0;
  A[2 * 6 + 4] = 0;
  A[2 * 6 + 5] = 1;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 6; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += Jpi[r * 3 + k] * A[k * 6 + c];
      }
      J_cam[r * 9 + c] = s;
    }
  }

  J_cam[0 * 9 + 6] = g * xn;
  J_cam[1 * 9 + 6] = g * yn;
  J_cam[0 * 9 + 7] = f * xn * r2;
  J_cam[1 * 9 + 7] = f * yn * r2;
  J_cam[0 * 9 + 8] = f * xn * r2 * r2;
  J_cam[1 * 9 + 8] = f * yn * r2 * r2;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += Jpi[r * 3 + k] * R[k * 3 + c];
      }
      J_point[r * 3 + c] = s;
    }
  }
}

static double huber_weight(double r2, double delta) {
  double r = sqrt(r2);
  if (delta <= 0.0 || r <= delta) return 1.0;
  return delta / r;
}

static double huber_cost(double r2, double delta) {
  double r = sqrt(r2);
  if (delta <= 0.0 || r <= delta) return 0.5 * r2;
  return delta * (r - 0.5 * delta);
}

static double compute_loss(const Camera *cams, const double *R_all, const Point *pts,
                           const Observation *obs, int no, double huber_delta) {
  double total = 0.0;
  for (int i = 0; i < no; ++i) {
    const Observation *o = &obs[i];
    const Camera *c = &cams[o->cam];
    const double *R = &R_all[o->cam * 9];
    double up, vp;
    project(c, R, pts[o->point].p, &up, &vp);
    double du = up - o->u;
    double dv = vp - o->v;
    total += huber_cost(du * du + dv * dv, huber_delta);
  }
  return total;
}

// Load BAL-format dataset: cameras, points, observations.
// Layout and parsing mirror msckf.c:load_bal exactly so the same
// datasets and downstream processing can be reused here.
static int load_bal(const char *path, Camera **cams_out, Point **pts_out, Observation **obs_out,
                    int *nc_out, int *np_out, int *no_out) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror("open dataset");
    return -1;
  }

  int nc = 0, np = 0, no = 0;
  if (fscanf(f, "%d %d %d", &nc, &np, &no) != 3) {
    fclose(f);
    return -1;
  }

  Observation *obs = (Observation *)malloc(sizeof(Observation) * (size_t)no);
  if (!obs) {
    fclose(f);
    return -1;
  }
  for (int i = 0; i < no; ++i) {
    if (fscanf(f, "%d %d %lf %lf", &obs[i].cam, &obs[i].point, &obs[i].u, &obs[i].v) != 4) {
      fclose(f);
      free(obs);
      return -1;
    }
  }

  Camera *cams = (Camera *)malloc(sizeof(Camera) * (size_t)nc);
  if (!cams) {
    fclose(f);
    free(obs);
    return -1;
  }
  for (int i = 0; i < nc; ++i) {
    Camera *c = &cams[i];
    if (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &c->aa[0], &c->aa[1], &c->aa[2], &c->t[0],
               &c->t[1], &c->t[2], &c->f, &c->k1, &c->k2) != 9) {
      fclose(f);
      free(cams);
      free(obs);
      return -1;
    }
  }

  Point *pts = (Point *)malloc(sizeof(Point) * (size_t)np);
  if (!pts) {
    fclose(f);
    free(cams);
    free(obs);
    return -1;
  }
  for (int i = 0; i < np; ++i) {
    if (fscanf(f, "%lf %lf %lf", &pts[i].p[0], &pts[i].p[1], &pts[i].p[2]) != 3) {
      fclose(f);
      free(pts);
      free(cams);
      free(obs);
      return -1;
    }
  }

  fclose(f);

  *cams_out = cams;
  *pts_out = pts;
  *obs_out = obs;
  *nc_out = nc;
  *np_out = np;
  *no_out = no;
  return 0;
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "data/dubrovnik/problem-16-22106-pre.txt";

  Camera *cams = NULL;
  Point *pts = NULL;
  Observation *obs = NULL;
  int nc = 0, np = 0, no = 0;

  if (load_bal(path, &cams, &pts, &obs, &nc, &np, &no) != 0) {
    fprintf(stderr, "shur: failed to load dataset %s\n", path);
    return 1;
  }

  int state_dim = 9 * nc;
  int valid_points = 0;
  double jacobian_eval_time = 0.0;
  double point_elim_time = 0.0;
  int jacobian_eval_calls = 0;
  int point_elim_calls = 0;
  int camera_number = nc;
  int point_number = np;
  int observation_number = no;

  // Precompute rotation matrices for all cameras
  double *R_all = (double *)malloc(sizeof(double) * 9 * (size_t)nc);
  if (!R_all) {
    fprintf(stderr, "shur: failed to allocate R_all\n");
    free(cams);
    free(pts);
    free(obs);
    return 1;
  }
  for (int i = 0; i < nc; ++i) {
    angle_axis_to_rot(cams[i].aa, &R_all[i * 9]);
  }

  const double huber_delta = 0.0; // match msckf default (no robustification)
  double residual_eval_time = 0.0;
  int residual_eval_calls = 0;
  int max_iters = 6;

  // Build point -> observation index lists (like msckf.c)
  PointObs *point_obs = (PointObs *)calloc((size_t)np, sizeof(PointObs));
  for (int i = 0; i < no; ++i) {
    point_obs[obs[i].point].count++;
  }
  for (int i = 0; i < np; ++i) {
    if (point_obs[i].count > 0) {
      point_obs[i].indices = (int *)malloc(sizeof(int) * point_obs[i].count);
      point_obs[i].count = 0;
    }
  }
  for (int i = 0; i < no; ++i) {
    int pid = obs[i].point;
    int idx = point_obs[pid].count++;
    point_obs[pid].indices[idx] = i;
  }

  for (int iter = 0; iter < max_iters; ++iter) {
    double t_res0 = wall_seconds();
    double loss_prev = compute_loss(cams, R_all, pts, obs, no, huber_delta);
    residual_eval_time += wall_seconds() - t_res0;
    residual_eval_calls++;
    printf("iter %d loss %.6f\n", iter, loss_prev);

    int row_cursor = 0;
    valid_points = 0;

    int *camera_index = (int *)malloc(sizeof(int) * nc);
    double *S = (double *)calloc((size_t)nc * state_dim * state_dim, sizeof(double));
    for (int pid = 0; pid < np; ++pid) {
      int cnt = point_obs[pid].count;
      if (cnt < 2) continue;
      valid_points++;
      int m = 2 * cnt;
      double *Hf = (double *)calloc((size_t)m * 3, sizeof(double));
      double *Hx = (double *)calloc((size_t)m * state_dim, sizeof(double));
      double *r = (double *)calloc((size_t)m, sizeof(double));
      double t_jac0 = wall_seconds();
      double Cp[9] = {0.0};
      double *A_camera = (double*)calloc((size_t)cnt * (state_dim * 3), sizeof(double));

      for (int j = 0; j < cnt; ++j) {
        const Observation *o = &obs[point_obs[pid].indices[j]];
        const Camera *c = &cams[o->cam];
        const double *R = &R_all[o->cam * 9];
        camera_index[j] = o->cam;

        double up, vp;
        project(c, R, pts[o->point].p, &up, &vp);
        double du = up - o->u;
        double dv = vp - o->v;
        double w = huber_weight(du * du + dv * dv, huber_delta);
        double sw = sqrt(w);
        double Jc[18], Jf[6];
        jacobians(c, R, pts[o->point].p, Jc, Jf);
        double Hc_local[18], Hp_local[6];
        for (int col = 0; col < 3; ++col) {
          Hp_local[0 * 3 + col] = sw * Jf[0 * 3 + col];
          Hp_local[1 * 3 + col] = sw * Jf[1 * 3 + col];
        }
        for (int col = 0; col < 9; ++col) {
          // For this toy Schur example we only keep the first 3 camera
          // columns (e.g. rotation), so copy those into Hc_local.
          if (col < 3) {
            Hc_local[0 * 3 + col] = sw * Jc[0 * 9 + col];
            Hc_local[1 * 3 + col] = sw * Jc[1 * 9 + col];
          }
        }
        r[2 * j + 0] = -sw * du;
        r[2 * j + 1] = -sw * dv;

        double temp_Cp[9];
        shur_construction<2, 3, 3>(Hc_local, Hp_local, temp_Cp);
        for (int i = 0; i < 9; ++i) {
          Cp[i] += temp_Cp[i];
        }

        double *A_camera_row = &A_camera[j * (state_dim * 3)];
        shur_construction<2,6, 3>(Hc_local, Hp_local, A_camera_row);
      }
      
      double Cp_inv[9];
      inv_3x3(Cp, Cp_inv);

      for (int ci = 0; ci < cnt; ++ci) {
        for (int cj = 0; cj < cnt; ++cj) {
          int ci_camera_index = camera_index[ci];
          int cj_camera_index = camera_index[cj];
          double *A_ci = &A_camera[ci * (state_dim * 3)];
          double *A_cj = &A_camera[cj * (state_dim * 3)];
          double* S_temp = (double*)calloc(state_dim * state_dim, sizeof(double));
          mat3_mul(A_ci, Cp_inv, S_temp);
          mat3_mul(S_temp, A_cj, S_temp);
          for (int i = 0; i < 9; ++i) {
            S_temp[i] *= -1.0;
          }

          // S block (camera_i, camera_j) = S_temp

          for (int row = 0; row < state_dim; ++row) {
            for (int col = 0; col < state_dim; ++col) {
              S[((ci_camera_index * state_dim) + row) * state_dim + col] = S_temp[row * state_dim + col];
            }
          }

          
        }
      }


      jacobian_eval_time += wall_seconds() - t_jac0;
      jacobian_eval_calls++;

      double t_elim0 = wall_seconds();
      // NOTE: This is where a full Schur complement / Givens elimination
      // would be applied to (Hf, Hx, r). For now we only time the placeholder.
      (void)Hf;
      (void)Hx;
      (void)r;
      point_elim_time += wall_seconds() - t_elim0;
      point_elim_calls++;

      free(Hf);
      free(Hx);
      free(r);
    }
    free(camera_index);
  }

  printf("shur: loaded dataset '%s'\n", path);
  printf("  cameras      : %d\n", nc);
  printf("  points       : %d\n", np);
  printf("  observations : %d\n", no);

  // Free point observation index lists
  for (int i = 0; i < np; ++i) {
    free(point_obs[i].indices);
  }
  free(point_obs);

  free(R_all);
  free(cams);
  free(pts);
  free(obs);

  return 0;
}
