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
// Adaptive Levenberg-Marquardt experiment. Point elimination and recovery use
// the same C + mu*diag(C); camera regularization is mu*diag(B). The gain ratio
// compares actual objective reduction with the undamped linearized residual
// model. Accepted steps shrink damping; rejected trials enlarge it. No two-view
// rank-one shortcut is used. LAPACK solves the equilibrated camera system.

#include "linalg.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Cholesky factor L of a 3x3 SPD matrix. Reads the lower triangle of A. */
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

/* Solve the regularized 3-by-3 point normal equations in place. */
static void solve_point_system(const double *normal, double *rhs) {
  double L[9];
  if (cholesky_3x3_spd(normal, L) != 0) {
    std::fill_n(rhs, 3, 0.0);
    return;
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < i; ++j)
      rhs[i] -= L[3 * i + j] * rhs[j];
    rhs[i] /= L[3 * i + i];
  }
  for (int i = 2; i >= 0; --i) {
    for (int j = i + 1; j < 3; ++j)
      rhs[i] -= L[3 * j + i] * rhs[j];
    rhs[i] /= L[3 * i + i];
  }
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

#ifdef SHUR_USE_LAPACK
extern "C" {
void dpotrf_(const char *, const int *, double *, const int *, int *);
void dpotrs_(const char *, const int *, const int *, const double *, const int *,
             double *, const int *, int *);
}
#endif

// In-place Cholesky L (lower) in S, then solve L y = b, L^T x = y; x overwrites b.
// Reads only the lower triangle of the row-major SPD matrix S. Returns 0 on success.
static int cholesky_solve(double *S, double *b, int n) {
#ifdef SHUR_USE_LAPACK
  // Row-major lower storage is column-major upper storage of the same symmetric
  // matrix. LAPACK must read U; the unused row-major upper triangle is not valid.
  const char triangle = 'U';
  const int nrhs = 1;
  int info = 0;
  dpotrf_(&triangle, &n, S, &n, &info);
  if (info != 0) {
    fprintf(stderr, "shur: LAPACK dpotrf info=%d\n", info);
    return -1;
  }
  dpotrs_(&triangle, &n, &nrhs, S, &n, b, &n, &info);
  if (info != 0) {
    fprintf(stderr, "shur: LAPACK dpotrs info=%d\n", info);
    return -1;
  }
  return 0;
#else
  for (int j = 0; j < n; ++j) {
    double d = S[j * n + j];
    for (int k = 0; k < j; ++k)
      d -= S[j * n + k] * S[j * n + k];
    if (!(d > 0.0) || !isfinite(d)) {
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
#endif
}

/* Equilibrate camera coordinates before factorization. Preserve the original
 * system because Cholesky overwrites it even when a pivot fails. A diagonal
 * shift in scaled coordinates is relative damping in original coordinates. */
static int scaled_camera_solve(double *S, double *b, int n) {
  std::vector<double> scale(n), rhs(b, b + n);
  std::vector<double> original((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    const double diag = S[(size_t)i * n + i];
    if (!isfinite(diag)) return -1;
    scale[i] = 1.0 / sqrt(std::max(fabs(diag), 1e-30));
  }
  for (int i = 0; i < n; ++i) {
    rhs[i] *= scale[i];
    for (int j = 0; j <= i; ++j) {
      double value = (S[(size_t)i * n + j] * scale[i]) * scale[j];
      if (!isfinite(value)) return -1;
      original[(size_t)i * n + j] = value;
    }
  }
  double damping = 0.0;
  for (int attempt = 0; attempt < 14; ++attempt) {
    std::copy(original.begin(), original.end(), S);
    std::copy(rhs.begin(), rhs.end(), b);
    for (int i = 0; i < n; ++i) S[(size_t)i * n + i] += damping;
    if (cholesky_solve(S, b, n) == 0) {
      bool finite = true;
      for (int i = 0; i < n; ++i) {
        b[i] *= scale[i];
        finite = finite && isfinite(b[i]);
      }
      if (finite) {
        printf("  camera solve relative damping=%.9g attempts=%d\n", damping, attempt + 1);
        return 0;
      }
    }
    damping = damping == 0.0 ? 1e-10 : damping * 10.0;
  }
  return -1;
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
  Observation *observations;
} PointObs;

static double wall_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Thin Householder QR of the three point columns. Reflectors are stored
 * below the diagonal of Jp, then applied to the first three unit vectors. */
static void householder_point_basis(double *Jp, double *basis, int rows) {
  double tau[3] = {};
  for (int col = 0; col < 3; ++col) {
    double scale = 0.0;
    for (int row = col; row < rows; ++row)
      scale = std::max(scale, fabs(Jp[3 * row + col]));
    if (scale < 1e-15)
      continue;
    double inv_scale = 1.0 / scale;
    double squared = 0.0;
    for (int row = col; row < rows; ++row) {
      double entry = Jp[3 * row + col] * inv_scale;
      squared += entry * entry;
    }
    double pivot = Jp[3 * col + col] * inv_scale;
    double diagonal = -copysign(sqrt(squared), pivot);
    tau[col] = (diagonal - pivot) / diagonal;
    double inv_lead = inv_scale / (pivot - diagonal);
    for (int row = col + 1; row < rows; ++row)
      Jp[3 * row + col] *= inv_lead;

    for (int j = col + 1; j < 3; ++j) {
      double dot = Jp[3 * col + j];
      for (int row = col + 1; row < rows; ++row)
        dot += Jp[3 * row + col] * Jp[3 * row + j];
      dot *= tau[col];
      Jp[3 * col + j] -= dot;
      for (int row = col + 1; row < rows; ++row)
        Jp[3 * row + j] -= Jp[3 * row + col] * dot;
    }
  }

  std::fill_n(basis, (size_t)rows * 3, 0.0);
  for (int col = 2; col >= 0; --col) {
    basis[3 * col + col] = 1.0 - tau[col];
    for (int row = col + 1; row < rows; ++row)
      basis[3 * row + col] = -Jp[3 * row + col] * tau[col];
    for (int j = col + 1; j < 3; ++j) {
      double dot = 0.0;
      for (int row = col + 1; row < rows; ++row)
        dot += Jp[3 * row + col] * basis[3 * row + j];
      dot *= tau[col];
      basis[3 * col + j] = -dot;
      for (int row = col + 1; row < rows; ++row)
        basis[3 * row + j] -= Jp[3 * row + col] * dot;
    }
  }
}

/* Damped point elimination: use exactly the same point normal matrix as
 * back-substitution. These columns are not an orthonormal basis: I-U U^T
 * is the damped Schur weight, not an idempotent projector. */
static void point_column_basis(double *Jp, double *basis, int rows, double point_mu) {
  double normal[9] = {};
  for (int row = 0; row < rows; ++row) {
    const double *p = Jp + 3 * row;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j <= i; ++j)
        normal[3 * i + j] += p[i] * p[j];
  }
  const double trace = normal[0] + normal[4] + normal[8];
  for (int i = 0; i < 3; ++i)
    normal[3 * i + i] += point_mu * std::max(normal[3 * i + i], 1e-12);
  double L[9];
  if (!isfinite(trace) || cholesky_3x3_spd(normal, L) != 0) {
    fprintf(stderr, "shur: damped point factorization failed\n");
    exit(2);
  }
  const double inv0 = 1.0 / L[0];
  const double inv1 = 1.0 / L[4];
  const double inv2 = 1.0 / L[8];
  for (int row = 0; row < rows; ++row) {
    const double *p = Jp + 3 * row;
    double *q = basis + 3 * row;
    q[0] = p[0] * inv0;
    q[1] = (p[1] - L[3] * q[0]) * inv1;
    q[2] = (p[2] - L[6] * q[0] - L[7] * q[1]) * inv2;
  }
}

static inline void add_camera_row(double *__restrict output, const double *__restrict right,
                                  double u, double v, int columns) {
  for (int j = 0; j < columns; ++j)
    output[j] += u * right[j] + v * right[9 + j];
}

/* Separate coefficient evaluation from the matrix writes so the compiler can
 * vectorize both fixed-size phases without cross-row alias checks. */
static inline void add_camera_pair(double *__restrict matrix, int stride,
                                   const double *__restrict left,
                                   const double *__restrict right,
                                   const double (&gram)[2][2]) {
  double row_u[9], row_v[9];
  for (int i = 0; i < 9; ++i) {
    row_u[i] = left[i] * gram[0][0] + left[9 + i] * gram[1][0];
    row_v[i] = left[i] * gram[0][1] + left[9 + i] * gram[1][1];
  }
  for (int i = 0; i < 9; ++i) {
    const double u = row_u[i], v = row_v[i];
    double *row = matrix + i * stride;
    row[0] += u * right[0] + v * right[9];
    row[1] += u * right[1] + v * right[10];
    row[2] += u * right[2] + v * right[11];
    row[3] += u * right[3] + v * right[12];
    row[4] += u * right[4] + v * right[13];
    row[5] += u * right[5] + v * right[14];
    row[6] += u * right[6] + v * right[15];
    row[7] += u * right[7] + v * right[16];
    row[8] += u * right[8] + v * right[17];
  }
}

static inline void rotate_point(const double *R, const double *point, double *rotated) {
  for (int i = 0; i < 3; ++i)
    rotated[i] = R[3 * i] * point[0] + R[3 * i + 1] * point[1] + R[3 * i + 2] * point[2];
}

static void project(const Camera *cam, const double R[9], const double Pw[3], double *u, double *v) {
  double Xc[3];
  rotate_point(R, Pw, Xc);
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

static void linearize(const Camera *cam, const double R[9], const double Pw[3],
                      double J_cam[18], double J_point[6], double *u, double *v) {
  double Xc[3];
  double RPw[3];
  rotate_point(R, Pw, RPw);
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
  *u = cam->f * g * xn;
  *v = cam->f * g * yn;
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

  for (int row = 0; row < 2; ++row) {
    const double *d = Jpi + row * 3;
    double *camera = J_cam + row * 9;
    /* A left rotation perturbation gives RPw cross the image gradient. */
    camera[0] = RPw[1] * d[2] - RPw[2] * d[1];
    camera[1] = RPw[2] * d[0] - RPw[0] * d[2];
    camera[2] = RPw[0] * d[1] - RPw[1] * d[0];
    std::copy_n(d, 3, camera + 3);
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

static bool bal_space(char ch) {
  return ch == ' ' || (ch >= '\t' && ch <= '\r');
}

static bool bal_digit(char ch) {
  return ch >= '0' && ch <= '9';
}

static bool read_short_integer(const char *&next, int &value) {
  const char *p = next;
  while (bal_space(*p))
    ++p;
  if (!bal_digit(*p))
    return false;
  int result = 0;
  int digits = 0;
  do {
    if (++digits > 9)
      return false;
    result = 10 * result + (*p++ - '0');
  } while (bal_digit(*p));
  if (*p && !bal_space(*p))
    return false;
  value = result;
  next = p;
  return true;
}

/* Short decimal tokens can be rounded with one operation: the mantissa
 * and every power of ten through 10^22 are exact in binary64. */
static bool read_short_decimal(const char *&next, double &value) {
  const char *p = next;
  while (bal_space(*p))
    ++p;
  bool negative = *p == '-';
  if (*p == '+' || *p == '-')
    ++p;

  uint64_t mantissa = 0;
  int digits = 0;
  int exponent = 0;
  bool fraction = false;
  while (bal_digit(*p) || (*p == '.' && !fraction)) {
    if (*p == '.') {
      fraction = true;
      ++p;
      continue;
    }
    if (++digits > 15)
      return false;
    mantissa = mantissa * 10 + (*p++ - '0');
    exponent -= fraction;
  }
  if (digits == 0)
    return false;

  if (*p == 'e' || *p == 'E') {
    ++p;
    bool negative_exp = *p == '-';
    if (*p == '+' || *p == '-')
      ++p;
    if (!bal_digit(*p))
      return false;
    int parsed = 0;
    do {
      parsed = parsed * 10 + (*p++ - '0');
      if (parsed > 37)
        return false;
    } while (bal_digit(*p));
    exponent += negative_exp ? -parsed : parsed;
  }
  if ((*p && !bal_space(*p)) || exponent < -22 || exponent > 22)
    return false;

  static constexpr double powers[] = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22
  };
  double magnitude = exponent < 0 ? double(mantissa) / powers[-exponent]
                                  : double(mantissa) * powers[exponent];
  value = negative ? -magnitude : magnitude;
  next = p;
  return true;
}

// Load whitespace-delimited BAL data into a contiguous buffer for token parsing.
static int load_bal(const char *path, Camera **cams_out, Point **pts_out, Observation **obs_out,
                    int *nc_out, int *np_out, int *no_out) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror("open dataset");
    return -1;
  }
  std::string input;
  char chunk[64 * 1024];
  size_t bytes;
  while ((bytes = fread(chunk, 1, sizeof(chunk), f)) != 0)
    input.append(chunk, bytes);
  bool read_error = ferror(f) != 0;
  fclose(f);
  if (read_error)
    return -1;

  const char *next = input.c_str();
  auto read_int = [&next](int &value) {
    if (read_short_integer(next, value))
      return true;
    char *end;
    long number = strtol(next, &end, 10);
    if (end == next || number < std::numeric_limits<int>::min() ||
        number > std::numeric_limits<int>::max())
      return false;
    value = static_cast<int>(number);
    next = end;
    return true;
  };
  auto read_double = [&next](double &value) {
    if (read_short_decimal(next, value))
      return true;
    char *end;
    value = strtod(next, &end);
    if (end == next)
      return false;
    next = end;
    return true;
  };

  int nc, np, no;
  if (!read_int(nc) || !read_int(np) || !read_int(no) || nc < 0 || np < 0 || no < 0)
    return -1;

  Observation *obs = (Observation *)malloc(sizeof(Observation) * (size_t)no);
  if (!obs)
    return -1;
  for (int i = 0; i < no; ++i) {
    Observation &o = obs[i];
    if (!read_int(o.cam) || !read_int(o.point) || !read_double(o.u) || !read_double(o.v) ||
        o.cam < 0 || o.cam >= nc || o.point < 0 || o.point >= np) {
      free(obs);
      return -1;
    }
  }

  Camera *cams = (Camera *)malloc(sizeof(Camera) * (size_t)nc);
  if (!cams) {
    free(obs);
    return -1;
  }
  for (int i = 0; i < nc; ++i) {
    Camera *c = &cams[i];
    if (!read_double(c->aa[0]) || !read_double(c->aa[1]) || !read_double(c->aa[2]) ||
        !read_double(c->t[0]) || !read_double(c->t[1]) || !read_double(c->t[2]) ||
        !read_double(c->f) || !read_double(c->k1) || !read_double(c->k2)) {
      free(cams);
      free(obs);
      return -1;
    }
  }

  Point *pts = (Point *)malloc(sizeof(Point) * (size_t)np);
  if (!pts) {
    free(cams);
    free(obs);
    return -1;
  }
  for (int i = 0; i < np; ++i) {
    if (!read_double(pts[i].p[0]) || !read_double(pts[i].p[1]) || !read_double(pts[i].p[2])) {
      free(pts);
      free(cams);
      free(obs);
      return -1;
    }
  }

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
  int max_iters = 6;
  if (const char *value = getenv("SHUR_MAX_ITERS")) {
    char *end = nullptr;
    long parsed = strtol(value, &end, 10);
    if (end == value || *end || parsed < 1 || parsed > 1000) {
      fprintf(stderr, "SHUR_MAX_ITERS must be an integer in [1,1000]\n");
      return 1;
    }
    max_iters = (int)parsed;
  }

  /* Keep the input order for loss evaluation and a contiguous track order
   * for the two passes that repeatedly visit each point's observations. */
  std::vector<PointObs> point_obs(np);
  std::vector<Observation> observations_by_point(no);
  for (int i = 0; i < no; ++i)
    ++point_obs[obs[i].point].count;
  int offset = 0;
  for (PointObs &track : point_obs) {
    if (track.count)
      track.observations = observations_by_point.data() + offset;
    offset += track.count;
    track.count = 0;
  }
  for (int i = 0; i < no; ++i) {
    PointObs &track = point_obs[obs[i].point];
    track.observations[track.count++] = obs[i];
  }
  /* Size point scratch buffers once and reuse them across tracks and iterations.
   * A track can contain repeated observations from the same camera. */
  int max_track_size = 0;
  for (int i = 0; i < np; ++i)
    max_track_size = std::max(max_track_size, point_obs[i].count);
  const size_t max_rows = (size_t)2 * max_track_size;
  std::vector<double> point_jacobian(max_rows * 3);
  /* Linearizations are stored in track order for contiguous back-substitution. */
  std::vector<double> camera_jacobian((size_t)no * 18);
  std::vector<double> cached_point_jacobian((size_t)no * 6);
  std::vector<double> residual((size_t)no * 2);
  std::vector<double> point_basis(max_rows * 3);
  std::vector<int> camera_ids(max_track_size);

  double target_cost = 0.0;
  if (const char *value = getenv("SHUR_TARGET_COST")) {
    char *end = nullptr;
    target_cost = strtod(value, &end);
    if (end == value || *end || !isfinite(target_cost) || target_cost <= 0.0) {
      fprintf(stderr, "SHUR_TARGET_COST must be a finite positive number\n");
      return 1;
    }
  }
  double lm_mu = 1e-4, lm_nu = 2.0;
  double current_loss = 0.0;
  bool solver_failed = false;
  /* Per-point profiling is optional. Whole-iteration timers remain on. */
  const bool profile_points = getenv("SHUR_PROFILE_POINTS") != nullptr;
  const auto point_time = [profile_points]() {
    return profile_points ? wall_seconds() : 0.0;
  };

  for (int iter = 0; iter < max_iters; ++iter) {
    double t_iter_start = wall_seconds();

    /* Step 1: Evaluate loss at current state */
    double t_step1 = wall_seconds();
    if (iter == 0)
      current_loss = compute_loss(cams, R_all, pts, obs, no, huber_delta);
    double loss_prev = current_loss;
    double dt_step1 = wall_seconds() - t_step1;
    printf("iter %d loss %.6f\n", iter, loss_prev);

    /* Step 2: Allocate reduced system S (state_dim x state_dim), b (state_dim); zeroed */
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

    /* Step 3: Assemble damped point elimination in observation coordinates.
     * Each observation has two rows and a single 2-by-9 camera Jacobian block. */
    std::vector<double> camera_diagonal(state_dim, 0.0);
    double t_step3 = wall_seconds();
    double dt_3a = 0.0, dt_3b = 0.0, dt_3c = 0.0;
    size_t track_offset = 0;
    for (int pid = 0; pid < np; ++pid) {
      int cnt = point_obs[pid].count;
      if (cnt < 2) continue;
      int m = 2 * cnt;
      double *Hf = point_jacobian.data();
      double *Hx = camera_jacobian.data() + track_offset * 18;
      double *cached_Hf = cached_point_jacobian.data() + track_offset * 6;
      double *basis = point_basis.data();
      double *r = residual.data() + track_offset * 2;
      track_offset += cnt;
      int *cam_list = camera_ids.data();

      /* Step 3a: Linearize into point rows and separate camera blocks. */
      double t_3a = point_time();
      for (int j = 0; j < cnt; ++j) {
        const Observation *o = &point_obs[pid].observations[j];
        const Camera *c = &cams[o->cam];
        const double *R = &R_all[o->cam * 9];
        cam_list[j] = o->cam;
        double up, vp;
        double Jc[18], Jf[6];
        linearize(c, R, pts[pid].p, Jc, Jf, &up, &vp);
        double du = up - o->u;
        double dv = vp - o->v;
        double sw = sqrt(huber_weight(du * du + dv * dv, huber_delta));
        for (int col = 0; col < 6; ++col) {
          double value = sw * Jf[col];
          Hf[j * 6 + col] = value;
          cached_Hf[j * 6 + col] = value;
        }
        for (int col = 0; col < 18; ++col) {
          const double value = sw * Jc[col];
          Hx[j * 18 + col] = value;
          camera_diagonal[o->cam * 9 + col % 9] += value * value;
        }
        r[2 * j] = -sw * du;
        r[2 * j + 1] = -sw * dv;
      }
      /* Step 3b: Factor the damped point block for Schur elimination. */
      double t_3b = point_time();
      dt_3a += t_3b - t_3a;
      point_column_basis(Hf, basis, m, lm_mu);

      /* Step 3c: Form blocks of I - U U^T, U = Jp * chol(C + mu Dp)^(-T).
       * This is a damped Schur weight, not an idempotent projector. */
      double t_3c = point_time();
      dt_3b += t_3c - t_3b;
      double point_rhs[3] = {};
      for (int row = 0; row < m; ++row)
        for (int j = 0; j < 3; ++j)
          point_rhs[j] += basis[3 * row + j] * r[row];

      for (int k1 = 0; k1 < cnt; ++k1) {
        const double *camera = Hx + k1 * 18;
        double rhs[2] = {r[2 * k1], r[2 * k1 + 1]};
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 3; ++j)
            rhs[i] -= basis[6 * k1 + 3 * i + j] * point_rhs[j];

        double *camera_rhs = b + cam_list[k1] * 9;
        for (int j = 0; j < 9; ++j)
          camera_rhs[j] += camera[j] * rhs[0] + camera[9 + j] * rhs[1];

        for (int k2 = k1; k2 < cnt; ++k2) {
          int left = k1;
          int right = k2;
          if (cam_list[left] < cam_list[right])
            std::swap(left, right);
          const int g1 = cam_list[left] * 9;
          const int g2 = cam_list[right] * 9;
          double gram[2][2] = {};
          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
              double entry = (k1 == k2 && i == j) ? 1.0 : 0.0;
              for (int k = 0; k < 3; ++k)
                entry -= basis[6 * left + 3 * i + k] * basis[6 * right + 3 * j + k];
              gram[i][j] = entry;
            }
          }

          const double *jl = Hx + left * 18;
          const double *jr = Hx + right * 18;
          if (g1 != g2) {
            add_camera_pair(S + g1 * state_dim + g2, state_dim, jl, jr, gram);
          } else {
            for (int i = 0; i < 9; ++i) {
              double u = jl[i] * gram[0][0] + jl[9 + i] * gram[1][0];
              double v = jl[i] * gram[0][1] + jl[9 + i] * gram[1][1];
              double *matrix_row = S + (g1 + i) * state_dim + g2;
              add_camera_row(matrix_row, jr, u, v, i + 1);
              if (k1 != k2)
                for (int j = i; j < 9; ++j)
                  S[(g2 + j) * state_dim + g1 + i] += u * jr[j] + v * jr[9 + j];
            }
          }
        }
      }
      dt_3c += point_time() - t_3c;
    }

    double dt_step3 = wall_seconds() - t_step3;

    /* Step 4: Add camera damping mu * diag(B) before the camera solve. */
    for (int i = 0; i < state_dim; ++i)
      S[(size_t)i * state_dim + i] += lm_mu * std::max(camera_diagonal[i], 1e-12);

    /* Step 5: Solve S*dx_cam = b via Cholesky. */
    double t_step5 = wall_seconds();
    int solve_ok = scaled_camera_solve(S, b, state_dim);
    double dt_step5 = wall_seconds() - t_step5;
    if (solve_ok != 0) {
      solver_failed = true;
      fprintf(stderr, "shur: Cholesky solve failed at iter %d\n", iter);
      free(camera_index);
      free(S);
      free(b);
      break;
    }
    /* b now holds the camera increment. */

    /* Step 6: Compute dx_point per point (back-substitution). */
    double t_step6 = wall_seconds();
    double *dx_points = (double *)calloc((size_t)np * 3, sizeof(double));
    if (!dx_points) {
      free(camera_index);
      free(S);
      free(b);
      break;
    }
    track_offset = 0;
    for (int pid = 0; pid < np; ++pid) {
      int cnt = point_obs[pid].count;
      if (cnt < 2) continue;
      double normal[9] = {};
      double rhs[3] = {};
      for (int j = 0; j < cnt; ++j, ++track_offset) {
        const Observation &o = point_obs[pid].observations[j];
        const double *camera = camera_jacobian.data() + track_offset * 18;
        const double *point = cached_point_jacobian.data() + track_offset * 6;
        const double *dx_camera = b + o.cam * 9;
        for (int row = 0; row < 2; ++row) {
          double projected_step = 0.0;
          for (int k = 0; k < 9; ++k)
            projected_step += camera[9 * row + k] * dx_camera[k];
          double remaining = residual[track_offset * 2 + row] - projected_step;
          for (int i = 0; i < 3; ++i) {
            rhs[i] += point[3 * row + i] * remaining;
            for (int k = 0; k <= i; ++k)
              normal[3 * i + k] += point[3 * row + i] * point[3 * row + k];
          }
        }
      }
      for (int i = 0; i < 3; ++i)
        normal[3 * i + i] += lm_mu * std::max(normal[3 * i + i], 1e-12);
      solve_point_system(normal, rhs);
      std::copy_n(rhs, 3, dx_points + 3 * pid);
    }

    // Reduction in the unregularized linearized residual model.
    double predicted_reduction = 0.0;
    track_offset = 0;
    for (int pid = 0; pid < np; ++pid) {
      if (point_obs[pid].count < 2) continue;
      for (int j = 0; j < point_obs[pid].count; ++j, ++track_offset) {
        int cam = point_obs[pid].observations[j].cam;
        for (int row = 0; row < 2; ++row) {
          double jd = 0.0;
          for (int k = 0; k < 9; ++k)
            jd += camera_jacobian[track_offset * 18 + row * 9 + k] * b[cam * 9 + k];
          for (int k = 0; k < 3; ++k)
            jd += cached_point_jacobian[track_offset * 6 + row * 3 + k] * dx_points[pid * 3 + k];
          predicted_reduction += residual[track_offset * 2 + row] * jd - 0.5 * jd * jd;
        }
      }
    }

    double dt_step6 = wall_seconds() - t_step6;

    /* Step 7: Evaluate one LM trial and its actual/predicted reduction. */
    double t_step7 = wall_seconds();
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
    const int max_backtracks = 1;
    double gain_ratio = -1.0;
    int accepted = 0;
    for (int bt = 0; bt < max_backtracks; ++bt) {
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
      gain_ratio = predicted_reduction > 0.0 ? (loss_prev - loss_try) / predicted_reduction : -1.0;
      if (isfinite(loss_try) && loss_try < loss_prev && gain_ratio > 1e-4) {
        current_loss = loss_try;
        for (int i = 0; i < nc; ++i) {
          cams[i] = cams_try[i];
          for (int k = 0; k < 9; ++k) R_all[i * 9 + k] = R_try[i * 9 + k];
        }
        for (int pid = 0; pid < np; ++pid) pts[pid] = pts_try[pid];
        accepted = 1;
        break;
      }
    }
    printf("  lm mu=%.9g predicted=%.17g rho=%.9g\n", lm_mu, predicted_reduction, gain_ratio);
    if (accepted) {
      lm_mu *= std::max(1.0 / 3.0, 1.0 - pow(2.0 * gain_ratio - 1.0, 3.0));
      lm_nu = 2.0;
    } else {
      lm_mu *= lm_nu;
      lm_nu = std::min(1e12, 2.0 * lm_nu);
    }
    lm_mu = std::max(1e-16, std::min(1e16, lm_mu));
    printf("  step accepted=%d alpha=%.9g cost=%.17g\n", accepted, alpha, current_loss);
    free(cams_try);
    free(pts_try);
    free(R_try);
    free(dx_points);
    free(camera_index);
    free(S);
    free(b);

    double dt_step7 = wall_seconds() - t_step7;
    double dt_iter  = wall_seconds() - t_iter_start;
    printf("  iter %d timing (s): loss=%.4f  ", iter, dt_step1);
    if (profile_points)
      printf("[3a jacobian=%.4f  3b point_qr=%.4f  3c gram=%.4f]  ",
             dt_3a, dt_3b, dt_3c);
    printf("step3=%.4f  solve=%.4f  backsolve=%.4f  linesearch=%.4f  total=%.4f\n",
           dt_step3, dt_step5, dt_step6, dt_step7, dt_iter);
    if (target_cost > 0.0 && current_loss <= target_cost) {
      printf("TARGET_REACHED attempts=%d cost=%.17g target=%.17g\n", iter + 1, current_loss, target_cost);
      break;
    }
  }

  if (const char *output = getenv("SHUR_WRITE_BAL")) {
    FILE *f = fopen(output, "w");
    if (!f) return 4;
    fprintf(f, "%d %d %d\n", nc, np, no);
    for (int i = 0; i < no; ++i) fprintf(f, "%d %d %.17g %.17g\n", obs[i].cam, obs[i].point, obs[i].u, obs[i].v);
    for (int i = 0; i < nc; ++i) {
      for (int j = 0; j < 3; ++j) fprintf(f, "%.17g\n", cams[i].aa[j]);
      for (int j = 0; j < 3; ++j) fprintf(f, "%.17g\n", cams[i].t[j]);
      fprintf(f, "%.17g\n%.17g\n%.17g\n", cams[i].f, cams[i].k1, cams[i].k2);
    }
    for (int i = 0; i < np; ++i) for (int j = 0; j < 3; ++j) fprintf(f, "%.17g\n", pts[i].p[j]);
    if (fclose(f) != 0) return 4;
  }
  printf("CHECK_FINAL_LOSS %.17g\n", compute_loss(cams, R_all, pts, obs, no, huber_delta));
  printf("shur: loaded dataset '%s'\n", path);
  printf("  cameras      : %d\n", nc);
  printf("  points       : %d\n", np);
  printf("  observations : %d\n", no);

  // Free point observation index lists

  free(R_all);
  free(cams);
  free(pts);
  free(obs);

  return solver_failed || (target_cost > 0.0 && current_loss > target_cost) ? 2 : 0;
}
