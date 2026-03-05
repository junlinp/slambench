#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

double dot3(const double a[3], const double b[3]);
void cross3(const double a[3], const double b[3], double out[3]);
void mat3_mul_vec(const double R[9], const double v[3], double out[3]);
void mat3_mul(const double A[9], const double B[9], double out[9]);
void mat3_identity(double I[9]);
void mat3_expmap(const double w[3], double R[9]);
void mat3_transpose(const double A[9], double At[9]);
void angle_axis_to_rot(const double aa[3], double R[9]);

/* Givens rotation helpers */
void givens(double a, double b, double *c, double *s);
/* Apply Givens to two rows (r1,r2) across columns of a row-major matrix (rows x cols). */
void apply_givens_rows(double *M, int rows, int cols, int r1, int r2, double c, double s);
/* Apply Givens to two entries of a vector (rows length). */
static inline void apply_givens_vec(double *v, int r1, int r2, double c, double s) {
  double v1 = v[r1];
  double v2 = v[r2];
  v[r1] = c * v1 - s * v2;
  v[r2] = s * v1 + c * v2;
}

/* Least squares solve using Givens QR: A (m x n, m>=n, row-major), rhs b (length m).
 * On output b holds solution x (length n). Returns 0 on success.
 */
int qr_solve_givens_row_order(double *A, int m, int n, double *b);
int qr_solve_givens_col_order(double *A, int m, int n, double *b);
int qr_solve_givens(double *A, int m, int n, double *b);

/* Explicit storage/elimination variants for benchmarking. */
int qr_solve_givens_rm_row_order(double *A, int m, int n, double *b);
int qr_solve_givens_rm_col_order(double *A, int m, int n, double *b);
int qr_solve_givens_cm_row_order(double *A, int m, int n, double *b);
int qr_solve_givens_cm_col_order(double *A, int m, int n, double *b);

#ifdef __cplusplus
}
#endif

#endif
