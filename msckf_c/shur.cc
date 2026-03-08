// Schur-complement bundle adjustment for BAL-style problems (cameras + points).
//
// ALGORITHM OVERVIEW
// ------------------
// We minimize sum over observations of (reprojection error)^2. The normal equations
// are [B  A] [dx_cam]   [b_cam]
//     [A^T Cp] [dx_pt ] = [b_pt ],
// where B = J_cam^T J_cam, A = J_cam^T J_pt, Cp = J_pt^T J_pt (per-point block), and
// (b_cam, b_pt) = -J^T r. Eliminating dx_pt gives the reduced (Schur) system:
//
//   S * dx_cam = b_reduced,
//   S = B - A * Cp^{-1} * A^T,
//   b_reduced = b_cam - A * Cp^{-1} * b_pt.
//
// We accumulate S and b_reduced by looping over points: for each point we form
// its B block (camera-camera), Cp and A (camera-point), then add B to S and
// subtract A Cp^{-1} A^T from S, and add the RHS contribution. Then we solve
// (S + lambda*I)*dx_cam = b_reduced (Cholesky), update cameras, and repeat.
//
// DETAILED STEPS (per iteration)
// -------------------------------
// 1. Evaluate loss (reprojection cost) at current camera/point state.
// 2. Allocate reduced system: S (state_dim x state_dim), b (state_dim); zero both.
// 3. For each point p with >= 2 observations, use Givens QR to eliminate the 3 point columns (same
//    as MSCKF). This gives null-space rows Hx_null where the point Jacobian is zero. Accumulate:
//      S += Hx_null^T Hx_null   (always PSD — no catastrophic cancellation)
//      b += Hx_null^T r_null
//    Note: the explicit formula S = B - A Cp^{-1} A^T has catastrophic cancellation (B and A Cp^{-1} A^T
//    are both ~1e8 but their difference is ~1e4), causing S to be numerically indefinite and requiring
//    large Tikhonov regularization that kills Gauss-Newton convergence.
// 4. Regularize S: same per-state damping as MSCKF (anchor_lambda=1e-3 on first 6, reg_lambda=1e-6 on rest).
// 5. Solve S*dx_cam = b via Cholesky (in-place); b becomes dx_cam (same delta as MSCKF bred).
// 6. Compute dx_point per point using MSCKF point backsolve: re-linearize and QR-solve Hf*dx_pt = r - Hc*dx_cam.
// 7. Backtracking line search (same as MSCKF): apply alpha*dx with R_new = exp(alpha*dtheta)*R; accept first alpha that reduces loss.
// 8. Repeat from step 1 for max_iters.

#include "linalg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

template<int residual_dim, int camera_dim, int point_dim>
void shur_construction(double *Hc, double *Hp, double *A) {
    // A = Hc^T * Hp (camera_dim x point_dim)
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

/* Cp_block += Hp^T * Hp (Hp is residual_dim x 3, result 3x3). Used to accumulate point Hessian. */
static void accum_HptHp(const double *Hp, int residual_dim, double *Cp_block) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < residual_dim; ++k)
                s += Hp[k * 3 + r] * Hp[k * 3 + c];
            Cp_block[r * 3 + c] += s;
        }
    }
}

/* Cholesky factor L of 3x3 SPD A (row-major): A = L L^T, L lower. Returns 0 on success, -1 if not SPD. */
static int cholesky_3x3_spd(const double *A, double *L) {
  double d = A[0 * 3 + 0];
  if (d <= 0.0) return -1;
  L[0 * 3 + 0] = sqrt(d);
  L[0 * 3 + 1] = 0.0;
  L[0 * 3 + 2] = 0.0;

  L[1 * 3 + 0] = A[1 * 3 + 0] / L[0 * 3 + 0];
  d = A[1 * 3 + 1] - L[1 * 3 + 0] * L[1 * 3 + 0];
  if (d <= 0.0) return -1;
  L[1 * 3 + 1] = sqrt(d);
  L[1 * 3 + 2] = 0.0;

  L[2 * 3 + 0] = A[2 * 3 + 0] / L[0 * 3 + 0];
  L[2 * 3 + 1] = (A[2 * 3 + 1] - L[2 * 3 + 0] * L[1 * 3 + 0]) / L[1 * 3 + 1];
  d = A[2 * 3 + 2] - L[2 * 3 + 0] * L[2 * 3 + 0] - L[2 * 3 + 1] * L[2 * 3 + 1];
  if (d <= 0.0) return -1;
  L[2 * 3 + 2] = sqrt(d);
  return 0;
}

/* Invert 3x3 SPD A into A_inv (row-major) via Cholesky: A = L L^T, A_inv = L^{-T} L^{-1}. Returns 0 on success, -1 if not SPD. */
static int inv_3x3(const double *A, double *A_inv) {
  double L[9];
  if (cholesky_3x3_spd(A, L) != 0) {
    for (int i = 0; i < 9; ++i) A_inv[i] = 0.0;
    return -1;
  }
  /* Solve L Y = I then L^T A_inv = Y (column j of A_inv). */
  for (int j = 0; j < 3; ++j) {
    double y[3];
    for (int i = 0; i < 3; ++i) {
      double s = (i == j) ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) s -= L[i * 3 + k] * y[k];
      y[i] = s / L[i * 3 + i];
    }
    for (int i = 2; i >= 0; --i) {
      double s = y[i];
      for (int k = i + 1; k < 3; ++k) s -= L[k * 3 + i] * A_inv[k * 3 + j];
      A_inv[i * 3 + j] = s / L[i * 3 + i];
    }
  }
  return 0;
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

// C(9x9) = A(9x3) * B(3x9); row-major
static void mat_9x3_3x9(const double *A, const double *B, double *C) {
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k)
        s += A[i * 3 + k] * B[k * 9 + j];
      C[i * 9 + j] = s;
    }
  }
}

// C(3x9) = A(3x3) * B(3x9); row-major
static void mat_3x3_3x9(const double *A, const double *B, double *C) {
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 9; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k)
        s += A[i * 3 + k] * B[k * 9 + j];
      C[i * 9 + j] = s;
    }
  }
}

// C(9x9) += A(9x3) * B(3x9); B is row-major 3x9 (transpose of 9x3 stored as 3x9)
static void mat_add_9x3_3x9(const double *A, const double *B, double *C) {
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k)
        s += A[i * 3 + k] * B[k * 9 + j];
      C[i * 9 + j] += s;
    }
  }
}

/* Compute dot = x^T S y (S is n x n row-major). */
static double dot_S(double *S, const double *x, const double *y, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      sum += x[i] * S[i * n + j] * y[j];
  return sum;
}

/* Compute z = L^T x; L is lower triangular stored in S (row-major), overwrites L. */
static void lt_times_vec(const double *L, const double *x, double *z, int n) {
  for (int i = 0; i < n; ++i) {
    double s = 0.0;
    for (int j = i; j < n; ++j)
      s += L[j * n + i] * x[j];
    z[i] = s;
  }
}

// In-place Cholesky L (lower) in S, then solve L y = b, L^T x = y; x overwrites b.
// S is n x n symmetric positive definite, row-major. Returns 0 on success.
static int cholesky_solve(double *S, double *b, int n) {
  for (int j = 0; j < n; ++j) {
    double d = S[j * n + j];
    for (int k = 0; k < j; ++k)
      d -= S[j * n + k] * S[j * n + k];
    if (d <= 0.0) {
      fprintf(stderr, "[shur] LINE %d cholesky_solve: pivot d=%.6g <= 0 at j=%d (n=%d)\n", __LINE__, d, j, n);
      return -1;
    }
    d = sqrt(d);
    S[j * n + j] = d;
    double inv_d = 1.0 / d;
    for (int i = j + 1; i < n; ++i) {
      double v = S[i * n + j];
      for (int k = 0; k < j; ++k)
        v -= S[i * n + k] * S[j * n + k];
      S[i * n + j] = v * inv_d;
    }
  }
  for (int i = 0; i < n; ++i) {
    double v = b[i];
    for (int k = 0; k < i; ++k)
      v -= S[i * n + k] * b[k];
    b[i] = v / S[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {
    double v = b[i];
    for (int k = i + 1; k < n; ++k)
      v -= S[k * n + i] * b[k];
    b[i] = v / S[i * n + i];
  }
  return 0;
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

/* Loss = sum over observations of huber_cost(||e||^2), e = (up - u, vp - v).
 * Same projection and residuals as the normal equations; when huber_delta=0
 * this is 0.5 * sum(du^2 + dv^2), matching the quadratic minimized by the solver. */
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

  const int state_dim = 9 * nc;   /* total camera state: 9 per camera */
  const int state_dim_per_cam = 9;
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
    /* Step 1: Evaluate loss at current state */
    fprintf(stderr, "[shur] LINE %d iter=%d\n", __LINE__, iter);
    double t_res0 = wall_seconds();
    double loss_prev = compute_loss(cams, R_all, pts, obs, no, huber_delta);
    residual_eval_time += wall_seconds() - t_res0;
    residual_eval_calls++;
    printf("iter %d loss %.6f\n", iter, loss_prev);

    /* Step 2: Allocate reduced system S (state_dim x state_dim), b (state_dim); zeroed */
    fprintf(stderr, "[shur] LINE %d alloc S,b\n", __LINE__);
    valid_points = 0;
    int *camera_index = (int *)malloc(sizeof(int) * (size_t)nc);
    double *S = (double *)calloc((size_t)state_dim * state_dim, sizeof(double));
    double *b = (double *)calloc((size_t)state_dim, sizeof(double));
    if (!camera_index || !S || !b) {
      fprintf(stderr, "shur: alloc failed (S/b)\n");
      free(camera_index);
      free(S);
      free(b);
      break;
    }

    /* Step 3: For each point with >= 2 observations, eliminate the point using Givens QR (identical
     * to MSCKF) and accumulate S += Hx_null^T Hx_null as a Gram matrix (always PSD — avoids the
     * catastrophic cancellation in the explicit formula S = B - A Cp^{-1} A^T). */
    for (int pid = 0; pid < np; ++pid) {
      int cnt = point_obs[pid].count;
      if (cnt < 2) continue;
      valid_points++;
      int m = 2 * cnt;
      double *Hf = (double *)calloc((size_t)m * 3, sizeof(double));        /* m x 3 point Jacobian */
      double *Hx = (double *)calloc((size_t)m * state_dim, sizeof(double)); /* m x state_dim camera Jacobian */
      double *r  = (double *)calloc((size_t)m, sizeof(double));             /* m x 1 residual */
      if (!Hf || !Hx || !r) { free(Hf); free(Hx); free(r); break; }

      double t_jac0 = wall_seconds();

      /* Step 3a: Build Hf, Hx, r per observation. */
      for (int j = 0; j < cnt; ++j) {
        const Observation *o = &obs[point_obs[pid].indices[j]];
        const Camera *c = &cams[o->cam];
        const double *R = &R_all[o->cam * 9];
        double up, vp;
        project(c, R, pts[o->point].p, &up, &vp);
        double du = up - o->u;
        double dv = vp - o->v;
        double w = huber_weight(du * du + dv * dv, huber_delta);
        double sw = sqrt(w);
        double Jc[18], Jf[6];
        jacobians(c, R, pts[o->point].p, Jc, Jf);
        for (int col = 0; col < 3; ++col) {
          Hf[(2 * j + 0) * 3 + col] = sw * Jf[0 * 3 + col];
          Hf[(2 * j + 1) * 3 + col] = sw * Jf[1 * 3 + col];
        }
        int off = o->cam * 9;
        for (int col = 0; col < 9; ++col) {
          Hx[(2 * j + 0) * state_dim + (off + col)] = sw * Jc[0 * 9 + col];
          Hx[(2 * j + 1) * state_dim + (off + col)] = sw * Jc[1 * 9 + col];
        }
        r[2 * j + 0] = -sw * du;
        r[2 * j + 1] = -sw * dv;
      }

      jacobian_eval_time += wall_seconds() - t_jac0;
      jacobian_eval_calls++;

      /* Step 3b: Givens QR to zero out Hf below its diagonal — same as MSCKF.
       * This is equivalent to projecting onto the left null space of Hf (exact, no inversion). */
      double t_elim0 = wall_seconds();
      for (int col = 0; col < 3; ++col) {
        for (int row = m - 1; row > col; --row) {
          double a = Hf[col * 3 + col];
          double bv = Hf[row * 3 + col];
          double c_g, s_g;
          givens(a, bv, &c_g, &s_g);
          apply_givens_rows(Hf, m, 3, col, row, c_g, s_g);
          apply_givens_rows(Hx, m, state_dim, col, row, c_g, s_g);
          apply_givens_vec(r, col, row, c_g, s_g);
        }
      }
      point_elim_time += wall_seconds() - t_elim0;
      point_elim_calls++;

      /* Step 3c: Accumulate S += Hx_null^T Hx_null and b += Hx_null^T r_null from null rows.
       * Hx_null (rows 3..m-1) is the projection of Hx onto null(Hf^T); S is PSD by construction. */
      int keep_rows = m - 3;
      for (int rr = 0; rr < keep_rows; ++rr) {
        const double *hrow = &Hx[(3 + rr) * state_dim];
        double rv = r[3 + rr];
        for (int ci = 0; ci < state_dim; ++ci) {
          if (hrow[ci] == 0.0) continue;
          b[ci] += hrow[ci] * rv;
          for (int cj = ci; cj < state_dim; ++cj) {
            double v = hrow[ci] * hrow[cj];
            S[ci * state_dim + cj] += v;
            if (ci != cj) S[cj * state_dim + ci] += v;
          }
        }
      }

      free(Hf);
      free(Hx);
      free(r);
    }

    /* Step 4: Regularize S — same per-state damping as MSCKF (anchor + reg lambda).
     * S is now built as a Gram matrix (always PSD), so no large floor is needed. */
    fprintf(stderr, "[shur] LINE %d point loop done, regularize\n", __LINE__);
    const double anchor_lambda = 1e-3;
    const double reg_lambda = 1e-6;
    for (int i = 0; i < state_dim; ++i)
      S[i * state_dim + i] += (i < 6) ? anchor_lambda : reg_lambda;

    /* Step 5: Solve S*dx_cam = b via Cholesky. S is PSD by construction; solve should succeed. */
    fprintf(stderr, "[shur] LINE %d before cholesky_solve\n", __LINE__);
    int solve_ok = cholesky_solve(S, b, state_dim);
    fprintf(stderr, "[shur] LINE %d after cholesky_solve ok=%d\n", __LINE__, solve_ok);
    if (solve_ok != 0) {
      fprintf(stderr, "shur: Cholesky solve failed at iter %d\n", iter);
      free(camera_index);
      free(S);
      free(b);
      break;
    }
    /* b now holds dx_cam (same delta as MSCKF bred) */

    /* Step 6: Compute dx_point = Cp_inv * (JpTr - A^T dx_cam) per point (same as MSCKF delta_pts). */
    double *dx_points = (double *)calloc((size_t)np * 3, sizeof(double));
    if (!dx_points) {
      free(camera_index);
      free(S);
      free(b);
      break;
    }
    for (int pid = 0; pid < np; ++pid) {
      int cnt = point_obs[pid].count;
      if (cnt < 2) continue;
      double Cp[9] = {0.0};
      double JpTr[3] = {0.0, 0.0, 0.0};
      double *A_camera = (double *)calloc((size_t)cnt * state_dim_per_cam * 3, sizeof(double));
      int *cam_idx = (int *)malloc(sizeof(int) * (size_t)cnt);

      for (int j = 0; j < cnt; ++j) {
        const Observation *o = &obs[point_obs[pid].indices[j]];
        const Camera *c = &cams[o->cam];
        const double *R = &R_all[o->cam * 9];
        cam_idx[j] = o->cam;
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
          Hc_local[0 * 9 + col] = sw * Jc[0 * 9 + col];
          Hc_local[1 * 9 + col] = sw * Jc[1 * 9 + col];
        }
        JpTr[0] += Hp_local[0 * 3 + 0] * (-sw * du) + Hp_local[1 * 3 + 0] * (-sw * dv);
        JpTr[1] += Hp_local[0 * 3 + 1] * (-sw * du) + Hp_local[1 * 3 + 1] * (-sw * dv);
        JpTr[2] += Hp_local[0 * 3 + 2] * (-sw * du) + Hp_local[1 * 3 + 2] * (-sw * dv);
        accum_HptHp(Hp_local, 2, Cp);
        double *A_row = &A_camera[j * (state_dim_per_cam * 3)];
        shur_construction<2, 9, 3>(Hc_local, Hp_local, A_row);
      }

      double trace_cp = Cp[0] + Cp[4] + Cp[8];
      const double cp_eps = 1e-10 * (1.0 + (trace_cp > 0 ? trace_cp : 0.0));
      double Cp_reg[9];
      for (int i = 0; i < 9; ++i) Cp_reg[i] = Cp[i];
      Cp_reg[0] += cp_eps;
      Cp_reg[4] += cp_eps;
      Cp_reg[8] += cp_eps;
      double Cp_inv[9];
      inv_3x3(Cp_reg, Cp_inv);

      double rhs_pt[3] = {JpTr[0], JpTr[1], JpTr[2]};
      for (int j = 0; j < cnt; ++j) {
        const double *Aj = &A_camera[j * (state_dim_per_cam * 3)];
        const double *dx_cam_j = &b[cam_idx[j] * 9];
        for (int k = 0; k < 3; ++k) {
          double dot = 0.0;
          for (int i = 0; i < 9; ++i)
            dot += Aj[i * 3 + k] * dx_cam_j[i];
          rhs_pt[k] -= dot;
        }
      }
      dx_points[pid * 3 + 0] = Cp_inv[0] * rhs_pt[0] + Cp_inv[1] * rhs_pt[1] + Cp_inv[2] * rhs_pt[2];
      dx_points[pid * 3 + 1] = Cp_inv[3] * rhs_pt[0] + Cp_inv[4] * rhs_pt[1] + Cp_inv[5] * rhs_pt[2];
      dx_points[pid * 3 + 2] = Cp_inv[6] * rhs_pt[0] + Cp_inv[7] * rhs_pt[1] + Cp_inv[8] * rhs_pt[2];

      free(A_camera);
      free(cam_idx);
    }

    /* Step 7: Backtracking line search (same as MSCKF): apply alpha*dx, accept first alpha that reduces loss. */
    Camera *cams_try = (Camera *)malloc(sizeof(Camera) * (size_t)nc);
    Point *pts_try = (Point *)malloc(sizeof(Point) * (size_t)np);
    double *R_try = (double *)malloc(sizeof(double) * 9 * (size_t)nc);
    if (!cams_try || !pts_try || !R_try) {
      fprintf(stderr, "shur: alloc trial state failed\n");
      free(dx_points);
      free(camera_index);
      free(S);
      free(b);
      break;
    }
    double alpha = 1.0;
    const int max_backtracks = 12;
    int accepted = 0;
    for (int bt = 0; bt < max_backtracks && alpha > 1e-4; ++bt, alpha *= 0.5) {
      for (int i = 0; i < nc; ++i) {
        cams_try[i] = cams[i];
        double dtheta[3] = {alpha * b[i * 9 + 0], alpha * b[i * 9 + 1], alpha * b[i * 9 + 2]};
        double dt[3] = {alpha * b[i * 9 + 3], alpha * b[i * 9 + 4], alpha * b[i * 9 + 5]};
        double dR[9];
        mat3_expmap(dtheta, dR);
        double newR[9];
        mat3_mul(dR, &R_all[i * 9], newR);
        memcpy(&R_try[i * 9], newR, sizeof(double) * 9);
        cams_try[i].t[0] += dt[0];
        cams_try[i].t[1] += dt[1];
        cams_try[i].t[2] += dt[2];
        cams_try[i].f += alpha * b[i * 9 + 6];
        cams_try[i].k1 += alpha * b[i * 9 + 7];
        cams_try[i].k2 += alpha * b[i * 9 + 8];
        rot_to_angle_axis(&R_try[i * 9], cams_try[i].aa);
      }
      for (int pid = 0; pid < np; ++pid) {
        pts_try[pid] = pts[pid];
        pts_try[pid].p[0] += alpha * dx_points[pid * 3 + 0];
        pts_try[pid].p[1] += alpha * dx_points[pid * 3 + 1];
        pts_try[pid].p[2] += alpha * dx_points[pid * 3 + 2];
      }
      double loss_try = compute_loss(cams_try, R_try, pts_try, obs, no, huber_delta);
      if (loss_try < loss_prev) {
        for (int i = 0; i < nc; ++i) {
          cams[i] = cams_try[i];
          for (int k = 0; k < 9; ++k) R_all[i * 9 + k] = R_try[i * 9 + k];
        }
        for (int pid = 0; pid < np; ++pid) pts[pid] = pts_try[pid];
        accepted = 1;
        break;
      }
    }
    free(cams_try);
    free(pts_try);
    free(R_try);
    free(dx_points);
    free(camera_index);
    free(S);
    free(b);
    fprintf(stderr, "[shur] LINE %d iter done\n", __LINE__);
  }

  fprintf(stderr, "[shur] LINE %d after iter loop\n", __LINE__);
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
