# SlamBench BAL Benchmarks

Run SymForce, Ceres, GTSAM, and a minimal C MSCKF-style benchmark in one container. Everything needed (code, binaries, dataset) is baked into the image; mount the workspace only if you want the logs on the host.

```
git clone --recursive git@github.com:dvorak0/slambench.git
cd slambench
docker build -t slambench-dev -f .devcontainer/Dockerfile .
docker run --rm --cpuset-cpus="0" slambench-dev
```

Note: if you want the logs/results on the host, mount the workspace:  
`docker run --rm --cpuset-cpus="0" -v "$PWD":/workspace slambench-dev`

To drop into a shell instead of running the benchmarks, override the entrypoint:
`docker run --rm -it --entrypoint bash slambench-dev`

Outputs (saved in the container or mounted workspace):
- `symforce.log`, `ceres.log`, `gtsam.log`, `msckf.log`

## CPU Schur build options

The `shur_runner` camera-pair kernel uses portable fixed-size arithmetic. No
architecture-specific intrinsics are required. By default the C++ target uses
the compiler's normal Release flags. To also tune that target for the machine
that builds it, enable the optional native option:

```bash
cmake -S msckf_c -B build-shur-native -DCMAKE_BUILD_TYPE=Release -DSHUR_NATIVE_ARCH=ON
cmake --build build-shur-native --target shur_runner -j2
./build-shur-native/shur_runner data/dubrovnik/problem-16-22106-pre.txt
```

The existing `msckf_linalg` C helper already uses the repository's
`-march=native` Release flags; this option controls only the C++ target.
Native builds can use AVX2/FMA on a compatible x86 CPU. They may not run on older
CPUs, and fused operations can change intermediate optimization losses through
floating-point rounding. This option does not enable `-ffast-math`. For a build
that disables floating-point contraction, additionally pass
`-DCMAKE_CXX_FLAGS=-ffp-contract=off` to a compiler supporting that flag. Validate
solution quality and timing on the target machine.

Enable the optional dense camera Cholesky backend with an LP64 LAPACK library:

```bash
cmake -S msckf_c -B build-shur-lapack -DCMAKE_BUILD_TYPE=Release \
  -DSHUR_NATIVE_ARCH=ON -DSHUR_USE_LAPACK=ON -DBLA_VENDOR=OpenBLAS
cmake --build build-shur-lapack --target shur_runner -j2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GOTO_NUM_THREADS=1 \
  ./build-shur-lapack/shur_runner data/dubrovnik/problem-16-22106-pre.txt
```

`SHUR_USE_LAPACK` defaults to OFF. With it enabled, `DPOTRF` and `DPOTRS`
replace only the dense camera factorization and triangular solves. Coordinate
scaling, damping retries, general projector assembly, and point recovery remain
unchanged. Row-major lower matrix storage is passed as column-major upper
storage (`UPLO='U'`), so no explicit matrix transpose is needed. Damping remains
necessary for weakly constrained systems; LAPACK alone does not guarantee that
the input matrix is numerically positive definite. Library threading is
controlled by the caller, not forced by the executable. Validate final costs
because factorization rounding can change nonlinear optimization trajectories.

## Experimental adaptive LM solver

This worktree replaces fixed point damping and backtracking with adaptive
Levenberg-Marquardt. Both point elimination and recovery use
`C + mu * diag(C)`; the camera block uses `B + mu * diag(B)`, with tiny diagonal
floors. The resulting `I - U U^T` is a damped Schur weight, not an orthogonal
projector. The actual/predicted objective reduction controls trial acceptance
and the next damping value. The initial `mu=1e-4` is shared by all datasets.
The general track path is used throughout, without the two-view rank-one case.

`SHUR_MAX_ITERS` optionally sets the maximum trial count (1 through 1000;
default 6). For matched-cost experiments, `SHUR_TARGET_COST` stops after an
accepted state reaches the requested positive objective value. Failure to
reach that target before the trial limit produces a nonzero exit code.
`SHUR_WRITE_BAL` optionally exports the final state for diagnostic checks;
leave it unset in timed runs. The objective is half the sum of squared
reprojection residuals, without a robust loss.

For example, after building with the LAPACK options above:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 GOTO_NUM_THREADS=1 \
  SHUR_MAX_ITERS=60 SHUR_TARGET_COST=286743.4 \
  ./build-shur-lapack/shur_runner path/to/problem-88-64298-pre.txt
```

Pin the enclosing Docker container with `--cpuset-cpus=0` for the single logical
CPU protocol. A reference final cost is a benchmark stopping criterion, not a
replacement for general convergence tolerances. Repeated tests on more datasets
are needed before treating this experimental solver as a general replacement.
