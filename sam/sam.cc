#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "../msckf_c/linalg.h"
}

struct Camera {
  double aa[3];
  double t[3];
  double f;
  double k1;
  double k2;
};

struct Point {
  double p[3];
};

struct Observation {
  int cam;
  int point;
  double u;
  double v;
};

static void skew(const double v[3], double S[9]) {
  S[0] = 0.0;
  S[1] = -v[2];
  S[2] = v[1];
  S[3] = v[2];
  S[4] = 0.0;
  S[5] = -v[0];
  S[6] = -v[1];
  S[7] = v[0];
  S[8] = 0.0;
}

static void project(const Camera *cam, const double R[9], const double Pw[3], double *u, double *v) {
  double Xc[3];
  mat3_mul_vec(R, Pw, Xc);
  Xc[0] += cam->t[0];
  Xc[1] += cam->t[1];
  Xc[2] += cam->t[2];
  const double invz = 1.0 / Xc[2];
  const double xn = -Xc[0] * invz;
  const double yn = -Xc[1] * invz;
  const double r2 = xn * xn + yn * yn;
  const double distortion = 1.0 + cam->k1 * r2 + cam->k2 * r2 * r2;
  *u = cam->f * distortion * xn;
  *v = cam->f * distortion * yn;
}

static void jacobians(const Camera *cam, const double R[9], const double Pw[3], double J_cam[18],
                      double J_point[6]) {
  double Xc[3];
  double RPw[3];
  mat3_mul_vec(R, Pw, RPw);
  Xc[0] = RPw[0] + cam->t[0];
  Xc[1] = RPw[1] + cam->t[1];
  Xc[2] = RPw[2] + cam->t[2];
  const double invz = 1.0 / Xc[2];
  const double xn = -Xc[0] * invz;
  const double yn = -Xc[1] * invz;

  const double r2 = xn * xn + yn * yn;
  const double g = 1.0 + cam->k1 * r2 + cam->k2 * r2 * r2;
  const double dg_dxn = cam->k1 * 2.0 * xn + cam->k2 * 4.0 * r2 * xn;
  const double dg_dyn = cam->k1 * 2.0 * yn + cam->k2 * 4.0 * r2 * yn;

  const double dpx_dxn = cam->f * (g + xn * dg_dxn);
  const double dpx_dyn = cam->f * (xn * dg_dyn);
  const double dpy_dxn = cam->f * (yn * dg_dxn);
  const double dpy_dyn = cam->f * (g + yn * dg_dyn);

  const double d_xn_dX = -invz;
  const double d_xn_dZ = Xc[0] * invz * invz;
  const double d_yn_dY = -invz;
  const double d_yn_dZ = Xc[1] * invz * invz;

  const double Jpi[6] = {
      dpx_dxn * d_xn_dX,
      dpx_dyn * d_yn_dY,
      dpx_dxn * d_xn_dZ + dpx_dyn * d_yn_dZ,
      dpy_dxn * d_xn_dX,
      dpy_dyn * d_yn_dY,
      dpy_dxn * d_xn_dZ + dpy_dyn * d_yn_dZ,
  };

  double S[9];
  skew(RPw, S);
  double A[18];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) A[r * 6 + c] = -S[r * 3 + c];
  }
  A[0 * 6 + 3] = 1.0;
  A[0 * 6 + 4] = 0.0;
  A[0 * 6 + 5] = 0.0;
  A[1 * 6 + 3] = 0.0;
  A[1 * 6 + 4] = 1.0;
  A[1 * 6 + 5] = 0.0;
  A[2 * 6 + 3] = 0.0;
  A[2 * 6 + 4] = 0.0;
  A[2 * 6 + 5] = 1.0;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 6; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += Jpi[r * 3 + k] * A[k * 6 + c];
      J_cam[r * 9 + c] = s;
    }
  }

  J_cam[0 * 9 + 6] = g * xn;
  J_cam[1 * 9 + 6] = g * yn;
  J_cam[0 * 9 + 7] = cam->f * xn * r2;
  J_cam[1 * 9 + 7] = cam->f * yn * r2;
  J_cam[0 * 9 + 8] = cam->f * xn * r2 * r2;
  J_cam[1 * 9 + 8] = cam->f * yn * r2 * r2;

  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 3; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += Jpi[r * 3 + k] * R[k * 3 + c];
      J_point[r * 3 + c] = s;
    }
  }
}

static double compute_cost(const std::vector<Camera> &cams, const std::vector<double> &R_all,
                           const std::vector<Point> &pts, const std::vector<Observation> &obs) {
  double total = 0.0;
  for (const auto &o : obs) {
    double up = 0.0, vp = 0.0;
    project(&cams[o.cam], &R_all[(size_t)o.cam * 9], pts[o.point].p, &up, &vp);
    const double du = up - o.u;
    const double dv = vp - o.v;
    total += 0.5 * (du * du + dv * dv);
  }
  return total;
}

static int load_bal(const char *path, std::vector<Camera> &cams, std::vector<Point> &pts,
                    std::vector<Observation> &obs) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;

  int nc = 0, np = 0, no = 0;
  if (fscanf(f, "%d %d %d", &nc, &np, &no) != 3) {
    fclose(f);
    return -1;
  }

  obs.resize((size_t)no);
  for (int i = 0; i < no; ++i) {
    if (fscanf(f, "%d %d %lf %lf", &obs[(size_t)i].cam, &obs[(size_t)i].point, &obs[(size_t)i].u,
               &obs[(size_t)i].v) != 4) {
      fclose(f);
      return -1;
    }
  }

  cams.resize((size_t)nc);
  for (int i = 0; i < nc; ++i) {
    Camera &c = cams[(size_t)i];
    if (fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &c.aa[0], &c.aa[1], &c.aa[2], &c.t[0], &c.t[1],
               &c.t[2], &c.f, &c.k1, &c.k2) != 9) {
      fclose(f);
      return -1;
    }
  }

  pts.resize((size_t)np);
  for (int i = 0; i < np; ++i) {
    if (fscanf(f, "%lf %lf %lf", &pts[(size_t)i].p[0], &pts[(size_t)i].p[1], &pts[(size_t)i].p[2]) != 3) {
      fclose(f);
      return -1;
    }
  }

  fclose(f);
  return 0;
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "data/dubrovnik/problem-16-22106-pre_stride=20.txt";
  int max_iters = (argc > 2) ? std::atoi(argv[2]) : 6;
  double damping = (argc > 3) ? std::atof(argv[3]) : 1e-6;

  std::vector<Camera> cams;
  std::vector<Point> pts;
  std::vector<Observation> obs;
  if (load_bal(path, cams, pts, obs) != 0) {
    std::fprintf(stderr, "[sam] failed to load dataset: %s\n", path);
    return 1;
  }

  const int nc = (int)cams.size();
  const int np = (int)pts.size();
  const int no = (int)obs.size();
  const int nvars = 9 * nc + 3 * np;
  const int nres = 2 * no;

  std::vector<double> R_all((size_t)nc * 9);
  for (int i = 0; i < nc; ++i) angle_axis_to_rot(cams[(size_t)i].aa, &R_all[(size_t)i * 9]);

  std::printf("[sam] loaded dataset '%s'\n", path);
  std::printf("[sam] cameras=%d points=%d observations=%d vars=%d residuals=%d\n", nc, np, no, nvars, nres);
  std::printf("[sam] pipeline: jacobian -> normal equations -> symcol(AMD) -> LDL\n");
  const double cost_initial = compute_cost(cams, R_all, pts, obs);
  const double t0 = (double)clock() / CLOCKS_PER_SEC;

  for (int iter = 0; iter < max_iters; ++iter) {
    std::vector<Eigen::Triplet<double>> J_triplets;
    J_triplets.reserve((size_t)no * 24);
    Eigen::VectorXd r(nres);

    for (int oi = 0; oi < no; ++oi) {
      const Observation &o = obs[(size_t)oi];
      const int row0 = 2 * oi;
      double up = 0.0, vp = 0.0;
      project(&cams[(size_t)o.cam], &R_all[(size_t)o.cam * 9], pts[(size_t)o.point].p, &up, &vp);
      r[row0 + 0] = up - o.u;
      r[row0 + 1] = vp - o.v;

      double Jc[18], Jp[6];
      jacobians(&cams[(size_t)o.cam], &R_all[(size_t)o.cam * 9], pts[(size_t)o.point].p, Jc, Jp);

      const int cam_col = 9 * o.cam;
      const int pt_col = 9 * nc + 3 * o.point;
      for (int c = 0; c < 9; ++c) {
        J_triplets.emplace_back(row0 + 0, cam_col + c, Jc[0 * 9 + c]);
        J_triplets.emplace_back(row0 + 1, cam_col + c, Jc[1 * 9 + c]);
      }
      for (int c = 0; c < 3; ++c) {
        J_triplets.emplace_back(row0 + 0, pt_col + c, Jp[0 * 3 + c]);
        J_triplets.emplace_back(row0 + 1, pt_col + c, Jp[1 * 3 + c]);
      }
    }

    Eigen::SparseMatrix<double> J(nres, nvars);
    J.setFromTriplets(J_triplets.begin(), J_triplets.end());

    // Build sparse normal equations: H = J^T J, g = J^T r.
    Eigen::SparseMatrix<double> H = J.transpose() * J;
    Eigen::VectorXd g = J.transpose() * r;
    for (int i = 0; i < nvars; ++i) H.coeffRef(i, i) += damping;
    H.makeCompressed();

    // "symcol": symbolic column ordering on H's sparsity pattern.
    Eigen::AMDOrdering<int> amd;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
    amd(H, P);

    // Numeric solve with sparse LDL^T.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt;
    ldlt.analyzePattern(H);
    ldlt.factorize(H);
    if (ldlt.info() != Eigen::Success) {
      std::fprintf(stderr, "[sam] LDL factorization failed at iter %d\n", iter);
      return 2;
    }
    Eigen::VectorXd dx = ldlt.solve(-g);
    if (ldlt.info() != Eigen::Success) {
      std::fprintf(stderr, "[sam] LDL solve failed at iter %d\n", iter);
      return 3;
    }

    for (int ci = 0; ci < nc; ++ci) {
      Camera &c = cams[(size_t)ci];
      double dR[9], Rnew[9];
      double dtheta[3] = {dx[9 * ci + 0], dx[9 * ci + 1], dx[9 * ci + 2]};
      mat3_expmap(dtheta, dR);
      mat3_mul(dR, &R_all[(size_t)ci * 9], Rnew);
      for (int k = 0; k < 9; ++k) R_all[(size_t)ci * 9 + k] = Rnew[k];
      rot_to_angle_axis(Rnew, c.aa);
      c.t[0] += dx[9 * ci + 3];
      c.t[1] += dx[9 * ci + 4];
      c.t[2] += dx[9 * ci + 5];
      c.f += dx[9 * ci + 6];
      c.k1 += dx[9 * ci + 7];
      c.k2 += dx[9 * ci + 8];
    }
    for (int pi = 0; pi < np; ++pi) {
      const int off = 9 * nc + 3 * pi;
      pts[(size_t)pi].p[0] += dx[off + 0];
      pts[(size_t)pi].p[1] += dx[off + 1];
      pts[(size_t)pi].p[2] += dx[off + 2];
    }

    const double cost = compute_cost(cams, R_all, pts, obs);
    std::printf("iter %d cost %.10e  |J|_nnz=%d  H_nnz=%d\n", iter, cost, (int)J.nonZeros(),
                (int)H.nonZeros());
  }
  const double cost_final = compute_cost(cams, R_all, pts, obs);
  const double t1 = (double)clock() / CLOCKS_PER_SEC;
  std::printf("initial cost: %.12e\n", cost_initial);
  std::printf("final cost: %.12e\n", cost_final);
  std::printf("iterations: %d\n", max_iters);
  std::printf("TIME %.6f\n", t1 - t0);
  std::printf("[sam] done\n");
  return 0;
}
