// shur_gpu_clean.cu — CUDA Schur complement for BAL bundle adjustment
// GPU: per-point Jacobian + Givens QR + Schur accumulation + backsolve
// GP106/Pascal (sm_61) — shared memory limit: 48KB per block
//
// KEY FIX: Hx (camera Jacobian) stored in GLOBAL memory, not shared memory
// Previous version: 112KB shared memory > 48KB limit → kernel never launched
// New version: ~2KB shared (Hf+r+cam_ids), Hx goes to device memory
//
// Huber loss: huber_delta > 0 enables robust weighting
// GPU target: GP106/Pascal (sm_61)

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { double r[3], t[3], f, k1, k2; } Camera;
typedef struct { double p[3]; } Point;
typedef struct { int cam, point; double u, v; } Observation;

static double wall_seconds() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__host__ __device__ void rodrigues_to_R(const double *r, double *R) {
  double th = sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
  if (th < 1e-12) { for(int i=0;i<9;i++) R[i]=0; R[0]=R[4]=R[8]=1; return; }
  double wx=r[0]/th, wy=r[1]/th, wz=r[2]/th, c=cos(th), s=sin(th), cm=1-c;
  R[0]=c+wx*wx*cm; R[1]=wx*wy*cm-wz*s; R[2]=wx*wz*cm+wy*s;
  R[3]=wx*wy*cm+wz*s; R[4]=c+wy*wy*cm; R[5]=wy*wz*cm-wx*s;
  R[6]=wx*wz*cm-wy*s; R[7]=wy*wz*cm+wx*s; R[8]=c+wz*wz*cm;
}

__device__ void d_project(double f, double k1, double k2, const double *R,
                          const double *t, const double *P, double *u, double *v) {
  double X=R[0]*P[0]+R[1]*P[1]+R[2]*P[2]+t[0];
  double Y=R[3]*P[0]+R[4]*P[1]+R[5]*P[2]+t[1];
  double Z=R[6]*P[0]+R[7]*P[1]+R[8]*P[2]+t[2];
  double iz=1.0/Z, xn=-X*iz, yn=-Y*iz;
  double r2=xn*xn+yn*yn, g=1+k1*r2+k2*r2*r2;
  // DEBUG: check projection
  if (isnan(f) || isnan(k1) || isnan(k2) || isnan(xn) || isnan(yn)) {
    printf("NAN in d_project: f=%.6e k1=%.6e k2=%.6e X=%.4f Y=%.4f Z=%.4f\n", f, k1, k2, X, Y, Z);
  }
  *u=f*xn*g; *v=f*yn*g;
}

__device__ void d_jacobians(double f, double k1, double k2, const double *R,
                            const double *t, const double *Pw, double *Jc, double *Jf) {
  double RPw[3];
  RPw[0]=R[0]*Pw[0]+R[1]*Pw[1]+R[2]*Pw[2];
  RPw[1]=R[3]*Pw[0]+R[4]*Pw[1]+R[5]*Pw[2];
  RPw[2]=R[6]*Pw[0]+R[7]*Pw[1]+R[8]*Pw[2];
  double Xc[3]={RPw[0]+t[0],RPw[1]+t[1],RPw[2]+t[2]};
  double iz=1.0/Xc[2], xn=-Xc[0]*iz, yn=-Xc[1]*iz;
  double r2=xn*xn+yn*yn, g=1+k1*r2+k2*r2*r2;
  double dgx=k1*2*xn+k2*4*r2*xn, dgy=k1*2*yn+k2*4*r2*yn;
  double dpxx=f*(g+xn*dgx), dpxy=f*(xn*dgy);
  double dpyx=f*(yn*dgx), dpyy=f*(g+yn*dgy);
  double dxX=-iz, dxZ=Xc[0]*iz*iz, dyY=-iz, dyZ=Xc[1]*iz*iz;
  double Jpi[6];
  Jpi[0]=dpxx*dxX; Jpi[1]=dpxy*dyY; Jpi[2]=dpxx*dxZ+dpxy*dyZ;
  Jpi[3]=dpyx*dxX; Jpi[4]=dpyy*dyY; Jpi[5]=dpyx*dxZ+dpyy*dyZ;
  Jc[0]=Jpi[1]*RPw[2]-Jpi[2]*RPw[1]; Jc[1]=Jpi[2]*RPw[0]-Jpi[0]*RPw[2]; Jc[2]=Jpi[0]*RPw[1]-Jpi[1]*RPw[0];
  Jc[9]=Jpi[4]*RPw[2]-Jpi[5]*RPw[1]; Jc[10]=Jpi[5]*RPw[0]-Jpi[3]*RPw[2]; Jc[11]=Jpi[3]*RPw[1]-Jpi[4]*RPw[0];
  for(int rr=0;rr<2;rr++) for(int cc=0;cc<3;cc++) Jc[rr*9+3+cc]=Jpi[rr*3+cc];
  Jc[6]=g*xn; Jc[15]=g*yn; Jc[7]=f*xn*r2; Jc[16]=f*yn*r2; Jc[8]=f*xn*r2*r2; Jc[17]=f*yn*r2*r2;
  for(int rr=0;rr<2;rr++) for(int cc=0;cc<3;cc++){double s=0;for(int k=0;k<3;k++)s+=Jpi[rr*3+k]*R[k*3+cc];Jf[rr*3+cc]=s;}
}

static double compute_loss(const Camera *cams, const double *R_all, const Point *pts,
                          const Observation *obs, int no) {
  double total=0;
  for (int i=0;i<no;i++) {
    const Camera *c=&cams[obs[i].cam]; const double *R=&R_all[obs[i].cam*9];
    const Point *P=&pts[obs[i].point];
    double X=R[0]*P->p[0]+R[1]*P->p[1]+R[2]*P->p[2]+c->t[0];
    double Y=R[3]*P->p[0]+R[4]*P->p[1]+R[5]*P->p[2]+c->t[1];
    double Z=R[6]*P->p[0]+R[7]*P->p[1]+R[8]*P->p[2]+c->t[2];
    double iz=1.0/Z, xn=-X*iz, yn=-Y*iz, r2=xn*xn+yn*yn, g=1+c->k1*r2+c->k2*r2*r2;
    double du=c->f*xn*g-obs[i].u, dv=c->f*yn*g-obs[i].v;
    total += du*du+dv*dv;
  }
  return total;
}

static void update_cameras(Camera *cams, double *R_all, const double *dx, int nc) {
  for (int i=0;i<nc;i++) {
    const double *d=&dx[i*9];
    double da=sqrt(d[3]*d[3]+d[4]*d[4]+d[5]*d[5]);
    double dR[9]={1,0,0,0,1,0,0,0,1};
    if (da>1e-12) {
      double wx=d[3]/da,wy=d[4]/da,wz=d[5]/da,cda=cos(da),sda=sin(da),cm=1-cda;
      dR[0]=cda+wx*wx*cm; dR[1]=wx*wy*cm-wz*sda; dR[2]=wx*wz*cm+wy*sda;
      dR[3]=wx*wy*cm+wz*sda; dR[4]=cda+wy*wy*cm; dR[5]=wy*wz*cm-wx*sda;
      dR[6]=wx*wz*cm-wy*sda; dR[7]=wy*wz*cm+wx*sda; dR[8]=cda+wz*wz*cm;
    }
    double R_old[9], R_new[9];
    memcpy(R_old,&R_all[i*9],72);
    for(int rr=0;rr<3;rr++) for(int cc=0;cc<3;cc++) R_new[rr*3+cc]=dR[rr*3]*R_old[cc]+dR[rr*3+1]*R_old[3+cc]+dR[rr*3+2]*R_old[6+cc];
    memcpy(&R_all[i*9],R_new,72);
    cams[i].t[0]+=d[0]; cams[i].t[1]+=d[1]; cams[i].t[2]+=d[2];
    cams[i].f+=d[6]; cams[i].k1+=d[7]; cams[i].k2+=d[8];
  }
}

static int cholesky_solve(double *A, double *b, int n) {
  for (int j=0;j<n;j++) {
    for (int k=0;k<j;k++) A[j*n+k]=0;
    double d=A[j*n+j]; for(int k=0;k<j;k++) d-=A[k*n+j]*A[k*n+j];
    if (d<=0) return -1;
    A[j*n+j]=sqrt(d);
    for(int i=j+1;i<n;i++) { for(int k=0;k<j;k++) A[j*n+i]-=A[k*n+j]*A[k*n+i]; A[j*n+i]/=A[j*n+j]; }
  }
  for(int i=0;i<n;i++) { for(int k=0;k<i;k++) b[i]-=A[k*n+i]*b[k]; b[i]/=A[i*n+i]; }
  for(int i=n-1;i>=0;i--) { for(int k=i+1;k<n;k++) b[i]-=A[k*n+i]*b[k]; b[i]/=A[i*n+i]; }
  return 0;
}

#define CUDA_CHK(call) do { cudaError_t e_ = call; if (e_ != cudaSuccess) fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); } while (0)

// Kernel 1: accumulate Schur complement S and b with Huber loss
//
// FIX: Hx stored in GLOBAL memory (d_Hx) instead of shared memory.
// Shared memory now only holds Hf + r + cam_ids (~2 KB, well under 48KB limit).
//
// huber_delta: 0 = squared error, >0 = Huber robust threshold
__global__ void accum_point_kernel(
    const Observation *d_obs, const Camera *d_cams, const Point *d_pts,
    const int *d_pt_off, const int *d_pt_cnt,
    const size_t *d_hx_off,  // per-point Hx offsets into d_Hx
    double *d_Hx,  // Hx stored in global memory (m x cc per point)
    double *d_S, double *d_b, int nc, int np, int sd, double huber_delta)
{
  int pid = blockIdx.x;
  if (pid >= np) return;
  int cnt = d_pt_cnt[pid];
  if (cnt < 2) return;
  int off = d_pt_off[pid];
  int m = 2*cnt, cc = cnt*9;

  // Shared memory: only Hf + r + cam_ids
  // Hf: m x 3  |  r: m  |  cam_ids: cnt
  // Max: 56*3 + 56 + 28 = 252 doubles = ~2 KB (well under 48KB)
  extern __shared__ double smem_[];
  double *Hf = smem_;
  double *r  = Hf + m*3;
  int *cam_ids = (int*)(r + m);

  // Point to this block's Hx region in global memory
  double *Hx_block = d_Hx + d_hx_off[pid];  // correctly offset per point

  double P[3];
  P[0]=d_pts[pid].p[0]; P[1]=d_pts[pid].p[1]; P[2]=d_pts[pid].p[2];

  // Phase 1: Build weighted Jacobians and residuals
  // Hf → shared, Hx → global, r → shared
  for (int j=threadIdx.x; j<cnt; j+=blockDim.x) {
    int oi=off+j; const Observation *o=&d_obs[oi]; const Camera *c=&d_cams[o->cam];
    cam_ids[j]=o->cam;
    double R[9]; rodrigues_to_R(c->r,R);
    double up,vp; d_project(c->f,c->k1,c->k2,R,c->t,P,&up,&vp);
    double du=up-o->u, dv=vp-o->v;
    // Huber weight
    double r2=du*du+dv*dv;
    double w = (huber_delta <= 0.0 || r2 <= huber_delta*huber_delta) ? 1.0 : huber_delta / sqrt(r2);
    double Jc[18],Jf[6]; d_jacobians(c->f,c->k1,c->k2,R,c->t,P,Jc,Jf);
    // DEBUG: print Jc for pid=1, first obs
    if (blockIdx.x == 1 && threadIdx.x == 0 && j == 0) {
        printf("GPU pid1 obs0: f=%.6f k1=%.6e k2=%.6e P=(%.4f,%.4f,%.4f)\n", c->f, c->k1, c->k2, P[0], P[1], P[2]);
        printf("GPU pid1 obs0: R[0..8]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n", R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8]);
        printf("GPU pid1 obs0: Jc[0..8]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n", Jc[0],Jc[1],Jc[2],Jc[3],Jc[4],Jc[5],Jc[6],Jc[7],Jc[8]);
        printf("GPU pid1 obs0: Jc[9..17]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n", Jc[9],Jc[10],Jc[11],Jc[12],Jc[13],Jc[14],Jc[15],Jc[16],Jc[17]);
        printf("GPU pid1 obs0: Jf[0..5]=(%.4f,%.4f,%.4f,%.4f,%.4f,%.4f)\n", Jf[0],Jf[1],Jf[2],Jf[3],Jf[4],Jf[5]);
    }
    // Hf: w * point Jacobian (2 rows x 3 cols) → shared
    for(int k=0;k<3;k++){ Hf[(2*j)*3+k]=Jf[k]*w; Hf[(2*j+1)*3+k]=Jf[3+k]*w; }
    // Hx: w * camera Jacobian (2 rows x 9 cols) → GLOBAL
    int coff=j*9;
    for(int k=0;k<9;k++){ Hx_block[(2*j)*cc+coff+k]=Jc[k]*w; Hx_block[(2*j+1)*cc+coff+k]=Jc[9+k]*w; }
    // Residual → shared
    r[2*j]=-du*w; r[2*j+1]=-dv*w;
  }
  __syncthreads();

  // DEBUG: print Hx right after Phase 1, before Givens QR
  if (blockIdx.x == 1 && threadIdx.x == 0) {
    printf("GPU pid=1 BEFORE GIVENS: Hx[0]=%.6e Hx[1]=%.6e Hx[cc]=%.6e cc=%d\n", Hx_block[0], Hx_block[1], Hx_block[cc], cc);
  }

  // Phase 2: Givens QR on Hf (m x 3) — thread 0 only
  // Hx stays in global memory; thread 0 reads, rotates, writes back
  if (threadIdx.x==0) {
    // DEBUG: print Hx before and after first rotation for pid=1
    if (blockIdx.x == 1) printf("GPU pid1 BEFORE ROT0: Hf[0]=%.6f Hf[5*3+0]=%.6f Hx[0]=%.6e Hx[27]=%.6e\n", Hf[0], Hf[5*3+0], Hx_block[0], Hx_block[27]);
    for(int col=0;col<3;col++){
      for(int row=m-1;row>col;row--){
        double a=Hf[col*3+col], bv=Hf[row*3+col];
        if(bv==0) continue;
        double h=sqrt(a*a+bv*bv), cg=a/h, sg=bv/h;
        if (blockIdx.x == 1 && col == 0 && row == 5) {
          double va0=Hx_block[col*cc+0], vb0=Hx_block[row*cc+0];
          printf("GPU rot col0 row5: cg=%.8f sg=%.8f Hx[0] before=%.6e after=%.6e\n", cg, sg, va0, cg*va0+sg*vb0);
        }
        // Rotate Hf (shared) — ALL columns, not just col..2
        for(int k=0;k<3;k++){
          double va=Hf[col*3+k], vb=Hf[row*3+k];
          Hf[col*3+k]=cg*va+sg*vb; Hf[row*3+k]=-sg*va+cg*vb;
        }
        // Rotate Hx rows (global): read → rotate → write
        for(int k=0;k<cc;k++){
          double va=Hx_block[col*cc+k], vb=Hx_block[row*cc+k];
          Hx_block[col*cc+k]=cg*va+sg*vb; Hx_block[row*cc+k]=-sg*va+cg*vb;
        }
        // Rotate r (shared)
        double ra=r[col], rb=r[row];
        r[col]=cg*ra+sg*rb; r[row]=-sg*ra+cg*rb;
      }
    }
  }
  __syncthreads();

  // Phase 3: Accumulate S and b from nullspace rows
  // Nullspace basis = rows 3..m-1 of Hx, stored in global memory
  int keep=m-3;
  const double *nb = Hx_block + 3*cc;  // &Hx_block[3*cc], nullspace [keep x cc]

  // DEBUG: print first and second point's Hx to compare with CPU mirror
  if (blockIdx.x <= 1 && threadIdx.x == 0) {
    printf("GPU pid=%d: cnt=%d m=%d cc=%d keep=%d\n", pid, cnt, m, cc, keep);
    printf("GPU Hx_block[0]=%.6e Hx_block[1]=%.6e Hx_block[cc]=%.6e\n", Hx_block[0], Hx_block[1], Hx_block[cc]);
    printf("GPU r[0]=%.6e r[1]=%.6e r[3]=%.6e\n", r[0], r[1], r[3]);
    printf("GPU nb[0]=%.6e nb[1]=%.6e nb[cc]=%.6e\n", nb[0], nb[1], nb[cc]);
  }

  // b: sum over null rows of nb[rr,col] * r[3+rr]
  for(int col=threadIdx.x; col<cc; col+=blockDim.x){
    int cid=cam_ids[col/9], param=col%9, gi=cid*9+param;
    double sum=0;
    for(int rr=0; rr<keep; rr++) sum += nb[rr*cc+col] * r[3+rr];
    atomicAdd(&d_b[gi], sum);
  }

  // S: symmetric accumulation from nullspace basis
  for(int k1=0;k1<cnt;k1++){
    int g1=cam_ids[k1]*9;
    for(int k2=k1;k2<cnt;k2++){
      int g2=cam_ids[k2]*9;
      for(int i=threadIdx.x; i<9; i+=blockDim.x){
        int jstart=(k2==k1)?i:0;
        for(int j=jstart; j<9; j++){
          double s=0;
          for(int rr=0; rr<keep; rr++) s += nb[rr*cc+k1*9+i] * nb[rr*cc+k2*9+j];
          atomicAdd(&d_S[(g1+i)*sd+(g2+j)], s);
          if(k1!=k2 || i!=j) atomicAdd(&d_S[(g2+j)*sd+(g1+i)], s);
        }
      }
    }
  }
}

// Kernel 2: backsolve for point updates
__global__ void backsolve_point_kernel(
    const Observation *d_obs, const Camera *d_cams, const Point *d_pts,
    const int *d_pt_off, const int *d_pt_cnt,
    const double *d_dx_cam, double *d_dx_pt, int np)
{
  int pid=blockIdx.x;
  if(pid>=np) return;
  int cnt=d_pt_cnt[pid];
  if(cnt<2) return;
  int off=d_pt_off[pid], m=2*cnt;
  extern __shared__ double sm2[];
  double *Hf=sm2, *r=Hf+m*3;
  int *cam_ids=(int*)(r+m);
  double P[3]; P[0]=d_pts[pid].p[0]; P[1]=d_pts[pid].p[1]; P[2]=d_pts[pid].p[2];
  for(int j=threadIdx.x; j<cnt; j+=blockDim.x){
    int oi=off+j; const Observation *o=&d_obs[oi]; const Camera *c=&d_cams[o->cam];
    cam_ids[j]=o->cam;
    double R[9]; rodrigues_to_R(c->r,R);
    double up,vp; d_project(c->f,c->k1,c->k2,R,c->t,P,&up,&vp);
    double du=up-o->u, dv=vp-o->v;
    double Jc[18],Jf[6]; d_jacobians(c->f,c->k1,c->k2,R,c->t,P,Jc,Jf);
    for(int k=0;k<3;k++){Hf[(2*j)*3+k]=Jf[k]; Hf[(2*j+1)*3+k]=Jf[3+k];}
    double ru=du, rv=dv;
    const double *dx=&d_dx_cam[o->cam*9];
    for(int k=0;k<9;k++){ru-=Jc[k]*dx[k]; rv-=Jc[9+k]*dx[k];}
    r[2*j]=-ru; r[2*j+1]=-rv;
  }
  __syncthreads();
  if(threadIdx.x==0){
    for(int col=0;col<3;col++){
      for(int row=m-1;row>col;row--){
        double a=Hf[col*3+col],bv=Hf[row*3+col];
        if(bv==0) continue;
        double h=sqrt(a*a+bv*bv),cg=a/h,sg=bv/h;
        for(int k=0;k<3;k++){
          double va=Hf[col*3+k],vb=Hf[row*3+k];
          Hf[col*3+k]=cg*va+sg*vb; Hf[row*3+k]=-sg*va+cg*vb;
        }
        double ra=r[col],rb=r[row];
        r[col]=cg*ra+sg*rb; r[row]=-sg*ra+cg*rb;
      }
    }
    for(int i=2;i>=0;i--){ for(int j=i+1;j<3;j++) r[i]-=Hf[i*3+j]*r[j]; r[i]/=Hf[i*3+i]; }
    d_dx_pt[pid*3]=r[0]; d_dx_pt[pid*3+1]=r[1]; d_dx_pt[pid*3+2]=r[2];
  }
}

int main(int argc, char **argv) {
  const char *path = (argc>1) ? argv[1] : "problem-16-22106-pre.txt";
  FILE *f = fopen(path, "r");
  if (!f) { perror(path); return 1; }
  int nc, np, no;
  if (fscanf(f, "%d %d %d", &nc, &np, &no) != 3) return 1;
  Camera *cams = (Camera*)calloc(nc, sizeof(Camera));
  Point *pts = (Point*)calloc(np, sizeof(Point));
  Observation *obs = (Observation*)malloc(no * sizeof(Observation));
  for (int i=0; i<no; i++) fscanf(f, "%d %d %lf %lf", &obs[i].cam, &obs[i].point, &obs[i].u, &obs[i].v);
  for (int i=0; i<nc; i++) fscanf(f, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
      &cams[i].r[0], &cams[i].r[1], &cams[i].r[2],
      &cams[i].t[0], &cams[i].t[1], &cams[i].t[2],
      &cams[i].f, &cams[i].k1, &cams[i].k2);
  for (int i=0; i<np; i++) fscanf(f, "%lf %lf %lf", &pts[i].p[0], &pts[i].p[1], &pts[i].p[2]);
  fclose(f);

  double *R_all = (double*)malloc(nc*9 * sizeof(double));
  for (int i=0; i<nc; i++) rodrigues_to_R(cams[i].r, &R_all[i*9]);

  int *pt_off = (int*)malloc(np * sizeof(int));
  int *pt_cnt = (int*)calloc(np, sizeof(int));
  int *tmp = (int*)calloc(np, sizeof(int));
  for (int i=0; i<no; i++) pt_cnt[obs[i].point]++;
  pt_off[0]=0;
  for (int p=1; p<np; p++) pt_off[p] = pt_off[p-1] + pt_cnt[p-1];

  // Compute Hx offsets per point: cumulative sum of (2*cnt)*(cnt*9) = 18*cnt^2
  size_t *hx_off = (size_t*)malloc((np+1) * sizeof(size_t));
  hx_off[0] = 0;
  for (int p=0; p<np; p++) {
    int m = 2*pt_cnt[p], cc = pt_cnt[p]*9;
    hx_off[p+1] = hx_off[p] + (size_t)m * cc;
  }
  size_t hx_total = hx_off[np];
  fprintf(stderr, "Total Hx global memory: %.1f MB\n", hx_total * sizeof(double) / 1e6);

  int sd = nc*9;
  double *S = (double*)calloc(sd*sd, sizeof(double));
  double *b = (double*)calloc(sd, sizeof(double));
  double *dx_pt = (double*)calloc(np*3, sizeof(double));

  Observation *d_obs; Camera *d_cams; Point *d_pts;
  int *d_pt_off, *d_pt_cnt;
  double *d_S, *d_b, *d_dx_cam, *d_dx_pt, *d_Hx;
  size_t *d_hx_off;
  CUDA_CHK(cudaMalloc(&d_obs, no * sizeof(Observation)));
  CUDA_CHK(cudaMalloc(&d_cams, nc * sizeof(Camera)));
  CUDA_CHK(cudaMalloc(&d_pts, np * sizeof(Point)));
  CUDA_CHK(cudaMalloc(&d_pt_off, np * sizeof(int)));
  CUDA_CHK(cudaMalloc(&d_pt_cnt, np * sizeof(int)));
  CUDA_CHK(cudaMalloc(&d_Hx, hx_total * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_hx_off, (np+1) * sizeof(size_t)));
  CUDA_CHK(cudaMemcpy(d_hx_off, hx_off, (np+1) * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMalloc(&d_S, sd*sd * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_b, sd * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dx_cam, sd * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dx_pt, np*3 * sizeof(double)));
  CUDA_CHK(cudaMemcpy(d_obs, obs, no * sizeof(Observation), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_pt_off, pt_off, np * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHK(cudaMemcpy(d_pt_cnt, pt_cnt, np * sizeof(int), cudaMemcpyHostToDevice));

  double huber_delta = 0.0;  // 0 = squared error; set >0 for Huber loss (e.g. 10.0)
  printf("shur_gpu: %d cams %d pts %d obs sd=%d loss=%.2f huber=%.1f\n",
         nc, np, no, sd, compute_loss(cams,R_all,pts,obs,no), huber_delta);

  double t0 = wall_seconds();

  for (int iter=0; iter<50; iter++) {
    CUDA_CHK(cudaMemcpy(d_cams, cams, nc*sizeof(Camera), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_pts, pts, np*sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_S, 0, sd*sd*sizeof(double)));
    CUDA_CHK(cudaMemset(d_b, 0, sd*sizeof(double)));

    // Shared memory: Hf(56*3) + r(56) + cam_ids(28) = 252+56+28 = 336 doubles = ~2.6 KB
    // Hx goes to global memory (d_Hx) to avoid 112KB shared memory overflow
    size_t smem_acc = (56*3 + 56 + 28) * sizeof(double);
    accum_point_kernel<<<np, 32, smem_acc>>>(
        d_obs, d_cams, d_pts, d_pt_off, d_pt_cnt, d_hx_off, d_Hx, d_S, d_b, nc, np, sd, huber_delta);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(S, d_S, sd*sd*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaMemcpy(b, d_b, sd*sizeof(double), cudaMemcpyDeviceToHost));

    if (iter == 0) {
      printf("DEBUG iter0: S[0,0]=%.6e S[1,1]=%.6e b[0]=%.6e b[1]=%.6e\n", S[0], S[1], b[0], b[1]);
      printf("DEBUG iter0: S[9,9]=%.6e b[9]=%.6e\n", S[9*sd+9], b[9]);
    }

    // Regularization
    for (int i=0; i<sd; i++) S[i*sd+i] += (i < 6) ? 1e-3 : 1e-6;
    if (cholesky_solve(S, b, sd) != 0) { fprintf(stderr, "Chol fail iter %d\n", iter); break; }

    double lp = compute_loss(cams, R_all, pts, obs, no);

    CUDA_CHK(cudaMemcpy(d_dx_cam, b, sd*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemset(d_dx_pt, 0, np*3*sizeof(double)));
    size_t smem_bs = (56*3 + 56 + 28) * sizeof(double);
    backsolve_point_kernel<<<np, 32, smem_bs>>>(
        d_obs, d_cams, d_pts, d_pt_off, d_pt_cnt, d_dx_cam, d_dx_pt, np);
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_CHK(cudaMemcpy(dx_pt, d_dx_pt, np*3*sizeof(double), cudaMemcpyDeviceToHost));

    Camera *ct2 = (Camera*)malloc(nc*sizeof(Camera));
    double *Rt2 = (double*)malloc(nc*9*sizeof(double));
    Point *pt2 = (Point*)malloc(np*sizeof(Point));
    int ok = 0;
    for (double a=1.0; a>=1.0/64; a*=0.5) {
      memcpy(ct2, cams, nc*sizeof(Camera));
      memcpy(Rt2, R_all, nc*9*sizeof(double));
      update_cameras(ct2, Rt2, b, nc);
      memcpy(pt2, pts, np*sizeof(Point));
      for (int p=0; p<np; p++) {
        pt2[p]=pts[p];
        pt2[p].p[0] += a*dx_pt[p*3];
        pt2[p].p[1] += a*dx_pt[p*3+1];
        pt2[p].p[2] += a*dx_pt[p*3+2];
      }
      double lt = compute_loss(ct2, Rt2, pt2, obs, no);
      if (lt < lp) {
        for (int i=0; i<nc; i++) cams[i] = ct2[i];
        memcpy(R_all, Rt2, nc*9*sizeof(double));
        for (int p=0; p<np; p++) pts[p] = pt2[p];
        ok = 1;
        printf("iter %d loss %.6f alpha=%.4f (%.3fs)\n", iter, lt, a, wall_seconds()-t0);
        break;
      }
    }
    free(ct2); free(Rt2); free(pt2);
    if (!ok) { printf("iter %d rejected\n", iter); break; }
  }
  printf("Total: %.3fs\n", wall_seconds()-t0);

  cudaFree(d_obs); cudaFree(d_cams); cudaFree(d_pts);
  cudaFree(d_pt_off); cudaFree(d_pt_cnt); cudaFree(d_hx_off); cudaFree(d_Hx);
  cudaFree(d_S); cudaFree(d_b); cudaFree(d_dx_cam); cudaFree(d_dx_pt);
  free(S); free(b); free(dx_pt); free(R_all); free(cams); free(pts); free(obs);
  free(pt_off); free(pt_cnt); free(tmp); free(hx_off);
  return 0;
}
