#include "linalg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

double dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

void mat3_mul_vec(const double R[9], const double v[3], double out[3]) {
  out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

void mat3_mul(const double A[9], const double B[9], double out[9]) {
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      out[r * 3 + c] =
          A[r * 3 + 0] * B[0 * 3 + c] + A[r * 3 + 1] * B[1 * 3 + c] + A[r * 3 + 2] * B[2 * 3 + c];
    }
  }
}

void mat3_identity(double I[9]) {
  memset(I, 0, sizeof(double) * 9);
  I[0] = I[4] = I[8] = 1.0;
}

void mat3_transpose(const double A[9], double At[9]) {
  At[0] = A[0];
  At[1] = A[1 * 3 + 0];
  At[2] = A[2 * 3 + 0];
  At[3] = A[0 * 3 + 1];
  At[4] = A[4];
  At[5] = A[2 * 3 + 1];
  At[6] = A[0 * 3 + 2];
  At[7] = A[1 * 3 + 2];
  At[8] = A[8];
}

static double norm3(const double v[3]) {
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void angle_axis_to_rot(const double aa[3], double R[9]) {
  double theta = norm3(aa);
  if (theta < 1e-12) {
    mat3_identity(R);
    return;
  }
  double axis[3] = {aa[0] / theta, aa[1] / theta, aa[2] / theta};
  double c = cos(theta);
  double s = sin(theta);
  double t = 1.0 - c;
  double x = axis[0], y = axis[1], z = axis[2];
  R[0] = t * x * x + c;
  R[1] = t * x * y - s * z;
  R[2] = t * x * z + s * y;
  R[3] = t * x * y + s * z;
  R[4] = t * y * y + c;
  R[5] = t * y * z - s * x;
  R[6] = t * x * z - s * y;
  R[7] = t * y * z + s * x;
  R[8] = t * z * z + c;
}

void rot_to_angle_axis(const double R[9], double aa[3]) {
  double trace = R[0] + R[4] + R[8];
  double cos_theta = (trace - 1.0) * 0.5;
  if (cos_theta >= 1.0) {
    aa[0] = 0.0;
    aa[1] = 0.0;
    aa[2] = 0.0;
    return;
  }
  if (cos_theta <= -1.0) {
    double x = (R[0] > -1.0) ? sqrt((R[0] + 1.0) * 0.5) : 0.0;
    double y = (R[4] > -1.0) ? sqrt((R[4] + 1.0) * 0.5) : 0.0;
    double z = (R[8] > -1.0) ? sqrt((R[8] + 1.0) * 0.5) : 0.0;
    if (x >= y && x >= z && x > 1e-12) {
      y = (R[1] + R[3]) / (4.0 * x);
      z = (R[2] + R[6]) / (4.0 * x);
    } else if (y >= z && y > 1e-12) {
      x = (R[1] + R[3]) / (4.0 * y);
      z = (R[5] + R[7]) / (4.0 * y);
    } else if (z > 1e-12) {
      x = (R[2] + R[6]) / (4.0 * z);
      y = (R[5] + R[7]) / (4.0 * z);
    } else {
      aa[0] = 0.0;
      aa[1] = 0.0;
      aa[2] = 0.0;
      return;
    }
    double theta = 3.14159265358979323846;
    aa[0] = theta * x;
    aa[1] = theta * y;
    aa[2] = theta * z;
    return;
  }
  double theta = acos(cos_theta);
  double s = sin(theta);
  if (s < 1e-12) {
    aa[0] = 0.0;
    aa[1] = 0.0;
    aa[2] = 0.0;
    return;
  }
  aa[0] = theta * (R[5] - R[7]) / (2.0 * s);
  aa[1] = theta * (R[6] - R[2]) / (2.0 * s);
  aa[2] = theta * (R[1] - R[3]) / (2.0 * s);
}

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

void mat3_expmap(const double w[3], double R[9]) {
  double theta = norm3(w);
  if (theta < 1e-12) {
    mat3_identity(R);
    return;
  }
  double wn[3] = {w[0] / theta, w[1] / theta, w[2] / theta};
  double K[9];
  skew(wn, K);
  double K2[9];
  mat3_mul(K, K, K2);
  double s = sin(theta);
  double c = cos(theta);
  double I[9];
  mat3_identity(I);
  for (int i = 0; i < 9; ++i) {
    R[i] = I[i] + s * K[i] + (1.0 - c) * K2[i];
  }
}

void givens(double a, double b, double *c, double *s) {
  if (fabs(b) < 1e-15) {
    *c = 1.0;
    *s = 0.0;
    return;
  }
  double r = hypot(a, b);
  *c = a / r;
  *s = -b / r;
}

void apply_givens_rows(double *M, int rows, int cols, int r1, int r2, double c, double s) {
  for (int col = 0; col < cols; ++col) {
    double x = M[r1 * cols + col];
    double y = M[r2 * cols + col];
    M[r1 * cols + col] = c * x - s * y;
    M[r2 * cols + col] = s * x + c * y;
  }
}

static int back_substitute_upper(const double *A, int n, double *b) {
  for (int i = n - 1; i >= 0; --i) {
    double sum = b[i];
    for (int j = i + 1; j < n; ++j) {
      sum -= A[i * n + j] * b[j];
    }
    double rii = A[i * n + i];
    if (fabs(rii) < 1e-12) return -1;
    b[i] = sum / rii;
  }
  return 0;
}

int qr_solve_givens_rm_row_order(double *A, int m, int n, double *b) {
  // In-place row-wise Givens QR on row-major A (m x n), m>=n.
  const double eps = 1e-15;
  for (int row = 0; row < m; ++row) {
    int pivots = row < n ? row : n;
    for (int i = 0; i < pivots; ++i) {
      double bval = A[row * n + i];
      if (fabs(bval) < eps) continue;
      double a = A[i * n + i];
      double c, s;
      givens(a, bval, &c, &s);

      for (int col = i; col < n; ++col) {
        double x = A[i * n + col];
        double y = A[row * n + col];
        A[i * n + col] = c * x - s * y;
        A[row * n + col] = s * x + c * y;
      }
      double bi = b[i];
      double br = b[row];
      b[i] = c * bi - s * br;
      b[row] = s * bi + c * br;
    }
  }
  return back_substitute_upper(A, n, b);
}

int qr_solve_givens_rm_col_order(double *A, int m, int n, double *b) {
  // In-place column-elimination order Givens QR on row-major A (m x n), m>=n.
  const double eps = 1e-15;
  for (int j = 0; j < n; ++j) {
    for (int r = j + 1; r < m; ++r) {
      double bval = A[r * n + j];
      if (fabs(bval) < eps) continue;
      double a = A[j * n + j];
      double c, s;
      givens(a, bval, &c, &s);

      for (int col = j; col < n; ++col) {
        double x = A[j * n + col];
        double y = A[r * n + col];
        A[j * n + col] = c * x - s * y;
        A[r * n + col] = s * x + c * y;
      }
      double bj = b[j];
      double br = b[r];
      b[j] = c * bj - s * br;
      b[r] = s * bj + c * br;
    }
  }
  return back_substitute_upper(A, n, b);
}

static int back_substitute_upper_col_major(const double *A, int m, int n, double *b) {
  (void)m;
  for (int i = n - 1; i >= 0; --i) {
    double sum = b[i];
    for (int j = i + 1; j < n; ++j) {
      sum -= A[j * m + i] * b[j];
    }
    double rii = A[i * m + i];
    if (fabs(rii) < 1e-12) return -1;
    b[i] = sum / rii;
  }
  return 0;
}

int qr_solve_givens_cm_row_order(double *A, int m, int n, double *b) {
  // In-place row-wise Givens QR on col-major A (m x n), m>=n.
  const double eps = 1e-15;
  for (int row = 0; row < m; ++row) {
    int pivots = row < n ? row : n;
    for (int i = 0; i < pivots; ++i) {
      double bval = A[i * m + row];
      if (fabs(bval) < eps) continue;
      double a = A[i * m + i];
      double c, s;
      givens(a, bval, &c, &s);

      for (int col = i; col < n; ++col) {
        double x = A[col * m + i];
        double y = A[col * m + row];
        A[col * m + i] = c * x - s * y;
        A[col * m + row] = s * x + c * y;
      }
      double bi = b[i];
      double br = b[row];
      b[i] = c * bi - s * br;
      b[row] = s * bi + c * br;
    }
  }
  return back_substitute_upper_col_major(A, m, n, b);
}

int qr_solve_givens_cm_col_order(double *A, int m, int n, double *b) {
  // In-place column-elimination order Givens QR on col-major A (m x n), m>=n.
  const double eps = 1e-15;
  for (int j = 0; j < n; ++j) {
    for (int r = j + 1; r < m; ++r) {
      double bval = A[j * m + r];
      if (fabs(bval) < eps) continue;
      double a = A[j * m + j];
      double c, s;
      givens(a, bval, &c, &s);

      for (int col = j; col < n; ++col) {
        double x = A[col * m + j];
        double y = A[col * m + r];
        A[col * m + j] = c * x - s * y;
        A[col * m + r] = s * x + c * y;
      }
      double bj = b[j];
      double br = b[r];
      b[j] = c * bj - s * br;
      b[r] = s * bj + c * br;
    }
  }
  return back_substitute_upper_col_major(A, m, n, b);
}

int qr_solve_givens_row_order(double *A, int m, int n, double *b) {
  return qr_solve_givens_rm_row_order(A, m, n, b);
}

int qr_solve_givens_col_order(double *A, int m, int n, double *b) {
  return qr_solve_givens_rm_col_order(A, m, n, b);
}

int qr_solve_givens(double *A, int m, int n, double *b) {
  // Default behavior kept for backward compatibility.
  return qr_solve_givens_rm_row_order(A, m, n, b);
}
