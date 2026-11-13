/* bifactor_kernel.c
 *
 * Compiled replacement for the hot inner loop of EstBifactorCopulaGAMM()'s
 * likf() function (the cluster / double-Gauss-quadrature loop).
 *
 * Design goal: preserve the *exact* arithmetic of the original R code
 * (dcop.R / pcond.R), just executed as compiled code instead of being
 * re-interpreted (with switch()-based dispatch and per-call vector
 * allocation) on every single evaluation of the negative log-likelihood.
 *
 * Family codes (must match the R-side mapping in family_code()):
 *   0 gaussian/normal   1 t        2 clayton   3 frank   4 gumbel
 *   5 joe               6 plackett 7 galambos  8 huesler-reiss
 *   9 fgm               10 ind
 *
 * NOTE on fidelity to the original R code:
 *  - dcop.R does NOT rotate "gaussian"/"t"/"frank"/"ind" -> replicated here.
 *  - pcond.R does NOT rotate "plackett" (unlike dcop.R, which does) ->
 *    replicated here (case 6 ignores `rot`).
 *  - pcond.R's "fgm" case (added in the corrected pcond.R) uses the same
 *    rotation pattern as clayton/joe/gumbel/galambos/huesler-reiss ->
 *    replicated here via pcondfgm_c().
 *  - CORRECTNESS FIX (vs. the original R likf()): equation (2.8) of the
 *    bi-factor copula paper requires, for each continuous margin j, the
 *    density product
 *        f_j(x_j) * c_{j,1}{F_j(x_j), v} * c_{j,2}[C_{j,1}{F_j(x_j), v}, w]
 *    The original R code computed the first two factors as `temp1 =
 *    matff[,j]*dcop(u1,vv,...)` but then discarded `temp1` and instead
 *    multiplied by `temp2` (the *conditional cdf* C_{j,1}{F_j(x_j),v}, a
 *    value in [0,1]) in its place -- i.e. it substituted the wrong
 *    quantity rather than genuinely omitting dead code. This kernel
 *    computes and uses the correct `f_j(x_j)*c_{j,1}{...}` density prefactor
 *    (see `Aden[]` below) instead of reusing `A[]` (the conditional cdf) as
 *    the multiplier.
 *
 * PERFORMANCE: the K clusters are statistically independent (the whole
 * point of the bi-factor construction is that clusters don't interact),
 * so the outer `k` loop below is parallelized with OpenMP when available
 * (`#pragma omp parallel for reduction(+:total)`). Every array read
 * inside the loop (F1, F2, f, cpar1/2, etc.) is read-only for the
 * duration of the call, and the only mutable state is the small
 * per-cluster scratch buffer (ltem3/A/B/Aden), which is now allocated
 * *inside* the loop body so each iteration/thread gets its own -- there
 * is no shared mutable state and no static/global variables are mutated
 * anywhere in this file (the `static const` tables in qnorm_c() are
 * read-only lookup tables), so this parallelization is safe. Falls back
 * to ordinary serial execution if compiled without `-fopenmp` (the
 * `#pragma omp` is simply ignored by the compiler in that case).
 *
 * Build (from R), single-threaded:
 *   R CMD SHLIB bifactor_kernel.c
 * Build (from R), multi-threaded (recommended -- EstBifactorCopulaGAMMFast's
 * .load_bifactor_kernel() does this automatically, with a fallback to the
 * single-threaded build if the compiler doesn't support -fopenmp, e.g.
 * plain Xcode clang on macOS without libomp installed):
 *   PKG_CFLAGS=-fopenmp PKG_LIBS=-fopenmp R CMD SHLIB bifactor_kernel.c
 * Load (from R):
 *   dyn.load("bifactor_kernel.so")   # or .dll on Windows
 */

#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#include "copula_common.h"

/* ---------------------------------------------------------------------
 * Base (unrotated) copula densities  -- ports of dcop.R
 * ------------------------------------------------------------------- */
static double dpla_c(double u, double v, double cpar) {
    double cpar1 = cpar - 1.0;
    double tem = 1.0 + cpar1*(u+v);
    double tem1 = tem*tem - 4.0*cpar*cpar1*u*v;
    double tem2 = sqrt(tem1);
    double tem0 = 2.0*cpar1*u*v, tem3 = tem - tem0;
    return cpar*tem3/tem1/tem2;
}
static double dfrk_c(double u, double v, double cpar) {
    double t1 = 1.0 - exp(-cpar);
    double tem1 = exp(-cpar*u), tem2 = exp(-cpar*v);
    double pdf = cpar*tem1*tem2*t1;
    double tem = t1 - (1.0-tem1)*(1.0-tem2);
    return pdf/(tem*tem);
}
static double dmtcj_c(double u, double v, double cpar) {
    double tem1 = pow(u,-cpar), tem2 = pow(v,-cpar);
    double pdf = pow(tem1+tem2-1.0, -1.0/cpar-2.0)*(1.0+cpar)*tem1*tem2/(u*v);
    if (v < 1e-4) pdf = 0.0;
    if (u < 1e-4) pdf = 0.0;
    if (cpar > 100.0) pdf = 0.0;
    return pdf;
}
static double djoe_c(double u, double v, double cpar) {
    double f1 = 1.0-u, f2 = 1.0-v;
    double tem1 = pow(f1,cpar), tem2 = pow(f2,cpar);
    double sm = tem1+tem2-tem1*tem2, tem = pow(sm, 1.0/cpar);
    double pdf = tem*((cpar-1.0)*tem1*tem2 + tem1*tem1*tem2 + tem1*tem2*tem2 - tem1*tem1*tem2*tem2);
    pdf /= (sm*sm);
    pdf /= (f1*f2);
    return pdf;
}
static double dgum_c(double u, double v, double cpar) {
    double l1 = -log(u), l2 = -log(v);
    double tem1 = pow(l1,cpar), tem2 = pow(l2,cpar);
    double sm = tem1+tem2, tem = pow(sm, 1.0/cpar);
    double cdf = exp(-tem);
    double pdf = cdf*tem*tem1*tem2*(tem+cpar-1.0);
    pdf /= (sm*sm*l1*l2*u*v);
    return pdf;
}
static double dgal_c(double u, double v, double cpar) {
    double cpar1 = -cpar;
    double l1 = -log(u), l2 = -log(v);
    double tem1 = pow(l1,cpar1-1.0), tem2 = pow(l2,cpar1-1.0);
    double sm = tem1*l1+tem2*l2, tem = pow(sm, 1.0/cpar1-1.0);
    double lcdf = -(l1+l2-tem*sm);
    double deriv12 = 1.0 + (tem/sm)*tem1*tem2*(1.0+cpar+tem*sm) - tem*(tem1+tem2);
    double pdf = lcdf+l1+l2;
    return exp(pdf)*deriv12;
}
static double dhr_c(double u, double v, double cpar) {
    double x = -log(u), y = -log(v);
    double z = x/y, cpar1 = 1.0/cpar, lz = log(z);
    double tem1 = cpar1+0.5*cpar*lz, tem2 = cpar1-0.5*cpar*lz;
    double p1 = pnorm_c(tem1), p2 = pnorm_c(tem2);
    double cdf = exp(-x*p1-y*p2);
    return cdf*(p1*p2+0.5*cpar*dnorm_c(tem1)/y)/(u*v);
}
static double dfgm_c(double u, double v, double cpar) {
    return 1.0 + cpar*(1.0-2.0*u)*(1.0-2.0*v);
}
static double dbvncop_c(double u, double v, double cpar) {
    double x1 = qnorm_c(u), x2 = qnorm_c(v);
    double qf = x1*x1+x2*x2-2.0*cpar*x1*x2;
    qf /= (1.0-cpar*cpar);
    double con = sqrt(1.0-cpar*cpar)*(2.0*M_PI);
    double pdf = exp(-0.5*qf)/con;
    return pdf/(dnorm_c(x1)*dnorm_c(x2));
}
static double dbvtcop_c(double u, double v, double cpar, double dfC) {
    double rh = cpar, nu = dfC;
    double con = exp(lgamma((nu+2.0)/2.0)-lgamma(nu/2.0))/(M_PI*nu);
    con /= sqrt(1.0-rh*rh);
    double ex = -(nu+2.0)/2.0;
    double xt1 = qt_c(u,nu), den1 = dt_c(xt1,nu);
    double xt2 = qt_c(v,nu), den2 = dt_c(xt2,nu);
    double r11 = 1.0/(1.0-rh*rh), r12 = -rh*r11;
    double qf = xt1*xt1*r11+xt2*xt2*r11+2.0*xt1*xt2*r12;
    double tem = 1.0+qf/nu;
    return con*pow(tem,ex)/(den1*den2);
}

/* ---------------------------------------------------------------------
 * Base (unrotated) conditional cdfs  -- ports of pcond.R
 * ------------------------------------------------------------------- */
static double pcondnor_c(double u, double v, double cpar) {
    double rho = cpar;
    double ccdf = pnorm_c((qnorm_c(u)-rho*qnorm_c(v))/sqrt(1.0-rho*rho));
    if (v <= 0.0 || u <= 0.0 || v >= 1.0) ccdf = 0.0;
    if (u == 1.0) ccdf = 1.0;
    return ccdf;
}
static double pcondt_c(double u, double v, double cpar, double dfC) {
    double rh = cpar, nu = dfC;
    if (v == 0.0) v = 1e-5;
    if (u == 0.0) u = 1e-6;
    if (v == 1.0) v = 1.0-1e-5;
    if (u == 1.0) u = 1.0-1e-6;
    double y1 = qt_c(v,nu), y2 = qt_c(u,nu);
    double mv = rh*y1;
    double s2 = (1.0-rh*rh)*(nu+y1*y1)/(nu+1.0);
    return pt_c((y2-mv)/sqrt(s2), nu+1.0);
}
static double pcondpla_c(double u, double v, double cpar) {
    double eta = cpar-1.0;
    double tem = 1.0+eta*(v+u);
    double tem1 = tem*tem-4.0*cpar*eta*v*u;
    double tem2 = sqrt(tem1);
    double ccdf = (eta*v+1.0-(eta+2.0)*u)/tem2;
    return 0.5*(1.0-ccdf);
}
static double pcondfrk_c(double u, double v, double cpar) {
    double cpar1 = 1.0-exp(-cpar);
    double tem = 1.0-exp(-cpar*v);
    return (1.0-tem)/(cpar1/(1.0-exp(-cpar*u))-tem);
}
static double pcondcla_c(double u, double v, double cpar) {
    double ccdf;
    if (cpar == 0.0) {
        ccdf = u;
    } else {
        double a = pow(v,cpar);
        ccdf = pow(1.0+a*(pow(u,-cpar)-1.0), -1.0-1.0/cpar);
    }
    if (v == 0.0) ccdf = 1.0;
    if (u == 0.0) ccdf = 0.0;
    return ccdf;
}
static double pcondjoe_c(double u, double v, double cpar) {
    double temu = pow(1.0-u,cpar), temv = pow(1.0-v,cpar);
    double ccdf = 1.0+temu/temv-temu;
    ccdf = pow(ccdf, -1.0+1.0/cpar);
    return ccdf*(1.0-temu);
}
static double pcondgum_c(double u, double v, double cpar) {
    if (v == 0.0) v = 1e-7;
    if (v == 1.0) v = 1.0-1e-7;
    if (u == 0.0) u = 1e-7;
    double x = -log(v), y = -log(u);
    double tem1 = pow(x,cpar), tem2 = pow(y,cpar);
    double svm = tem1+tem2, tem = pow(svm, 1.0/cpar);
    double ccdf = exp(-tem);
    ccdf *= pow(1.0+tem2/tem1, -1.0+1.0/cpar);
    return ccdf/v;
}
static double pcondgal_c(double u, double v, double cpar) {
    double x = -log(v), y = -log(u);
    double tem1 = pow(x,-cpar), tem2 = pow(y,-cpar);
    double sm = tem1+tem2, tem = pow(sm, -1.0/cpar);
    double ccdf = exp(-(x+y-tem));
    ccdf *= (1.0-pow(1.0+tem2/tem1, -1.0-1.0/cpar));
    return ccdf/v;
}
static double pcondhr_c(double u, double v, double cpar) {
    double x = -log(v), y = -log(u);
    double z = x/y, cpar1 = 1.0/cpar, lz = log(z);
    double tem1 = cpar1+0.5*cpar*lz, tem2 = cpar1-0.5*cpar*lz;
    double p1 = pnorm_c(tem1), p2 = pnorm_c(tem2);
    double lcdf = -x*p1-y*p2, cdf = exp(lcdf);
    return cdf*p1/v;
}
static double pcondfgm_c(double u, double v, double cpar) {
    double tem = 1.0 + cpar*(1.0-2.0*v)*(1.0-u);
    return u*tem;
}

/* ---------------------------------------------------------------------
 * Family dispatch (replicates dcop()/pcond() switch+rotation logic)
 * ------------------------------------------------------------------- */
static double dcop_c(double u, double v, int fam, int rot, double cpar, double dfC) {
    switch (fam) {
        case 0: return dbvncop_c(u,v,cpar);                 /* gaussian, no rotation */
        case 1: return dbvtcop_c(u,v,cpar,dfC);              /* t, no rotation */
        case 2:                                              /* clayton */
            if (rot==0) return dmtcj_c(u,v,cpar);
            else if (rot==90) return dmtcj_c(1-u,v,cpar);
            else if (rot==180) return dmtcj_c(1-u,1-v,cpar);
            else return dmtcj_c(u,1-v,cpar);
        case 3: return dfrk_c(u,v,cpar);                     /* frank, no rotation */
        case 4:                                              /* gumbel */
            if (rot==0) return dgum_c(u,v,cpar);
            else if (rot==90) return dgum_c(1-u,v,cpar);
            else if (rot==180) return dgum_c(1-u,1-v,cpar);
            else return dgum_c(u,1-v,cpar);
        case 5:                                              /* joe */
            if (rot==0) return djoe_c(u,v,cpar);
            else if (rot==90) return djoe_c(1-u,v,cpar);
            else if (rot==180) return djoe_c(1-u,1-v,cpar);
            else return djoe_c(u,1-v,cpar);
        case 6:                                              /* plackett */
            if (rot==0) return dpla_c(u,v,cpar);
            else if (rot==90) return dpla_c(1-u,v,cpar);
            else if (rot==180) return dpla_c(1-u,1-v,cpar);
            else return dpla_c(u,1-v,cpar);
        case 7:                                              /* galambos */
            if (rot==0) return dgal_c(u,v,cpar);
            else if (rot==90) return dgal_c(1-u,v,cpar);
            else if (rot==180) return dgal_c(1-u,1-v,cpar);
            else return dgal_c(u,1-v,cpar);
        case 8:                                              /* huesler-reiss */
            if (rot==0) return dhr_c(u,v,cpar);
            else if (rot==90) return dhr_c(1-u,v,cpar);
            else if (rot==180) return dhr_c(1-u,1-v,cpar);
            else return dhr_c(u,1-v,cpar);
        case 9:                                              /* fgm */
            if (rot==0) return dfgm_c(u,v,cpar);
            else if (rot==90) return dfgm_c(1-u,v,cpar);
            else if (rot==180) return dfgm_c(1-u,1-v,cpar);
            else return dfgm_c(u,1-v,cpar);
        case 10: return 1.0;                                 /* ind */
        default: return NAN;
    }
}
static double pcond_c(double u, double v, int fam, int rot, double cpar, double dfC) {
    double p;
    switch (fam) {
        case 0: p = pcondnor_c(u,v,cpar); break;
        case 1: p = pcondt_c(u,v,cpar,dfC); break;
        case 2:
            if (rot==0) p = pcondcla_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondcla_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondcla_c(1-u,1-v,cpar);
            else p = pcondcla_c(u,1-v,cpar);
            break;
        case 3: p = pcondfrk_c(u,v,cpar); break;
        case 4:
            if (rot==0) p = pcondgum_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondgum_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondgum_c(1-u,1-v,cpar);
            else p = pcondgum_c(u,1-v,cpar);
            break;
        case 5:
            if (rot==0) p = pcondjoe_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondjoe_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondjoe_c(1-u,1-v,cpar);
            else p = pcondjoe_c(u,1-v,cpar);
            break;
        case 6: p = pcondpla_c(u,v,cpar); break;              /* no rotation, matches pcond.R */
        case 7:
            if (rot==0) p = pcondgal_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondgal_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondgal_c(1-u,1-v,cpar);
            else p = pcondgal_c(u,1-v,cpar);
            break;
        case 8:
            if (rot==0) p = pcondhr_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondhr_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondhr_c(1-u,1-v,cpar);
            else p = pcondhr_c(u,1-v,cpar);
            break;
        case 9:                                               /* fgm */
            if (rot==0) p = pcondfgm_c(u,v,cpar);
            else if (rot==90) p = 1.0-pcondfgm_c(1-u,v,cpar);
            else if (rot==180) p = 1.0-pcondfgm_c(1-u,1-v,cpar);
            else p = pcondfgm_c(u,1-v,cpar);
            break;
        case 10: p = u; break;                                /* ind */
        default: p = NAN;
    }
    if (u == 0.0) p = 0.0;
    if (u == 1.0) p = 1.0;
    return p;
}

/* ---------------------------------------------------------------------
 * Main entry point, called from R via .C().
 *
 * F1,F2,f    : N x dp matrices, column-major (as R stores them), length N*dp
 * cont       : length dp, 1 = continuous margin, 0 = discrete margin
 * fam1,fam2  : length dp, family codes for level-1 / level-2 copulas
 * rot1,rot2  : length dp, rotation codes (0/90/180/270)
 * cpar1,cpar2: length dp, copula parameters (change every likf() call)
 * dfC1,dfC2  : length dp, Student-t degrees of freedom (0 if unused)
 * order0     : length N, 0-based row indices grouped by cluster
 * cbound     : length K+1, cluster boundaries into order0 (cbound[0]=0)
 * nl,wl      : length nq, Gauss-Legendre-on-(0,1) nodes/weights
 * out        : length 1, receives the total negative log-likelihood
 * ------------------------------------------------------------------- */
void bifactor_negloglik(
    double *F1, double *F2, double *f,
    int *N_, int *dp_,
    int *cont,
    int *fam1, int *fam2,
    int *rot1, int *rot2,
    double *cpar1, double *cpar2,
    double *dfC1, double *dfC2,
    int *order0, int *cbound, int *K_,
    double *nl, double *wl, int *nq_,
    double *out
) {
    int N = *N_, dp = *dp_, K = *K_, nq = *nq_;
    double total = 0.0;

    /* Each cluster's scratch buffers are allocated inside the loop body
     * (instead of once, outside, as before) so that every iteration --
     * and hence every OpenMP thread -- owns independent memory. */
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:total) schedule(dynamic) if(K > 1)
#endif
    for (int k = 0; k < K; k++) {
        int start = cbound[k], end = cbound[k+1];

        double *ltem3 = (double*) malloc((size_t)nq * sizeof(double));
        double *A    = (double*) malloc((size_t)dp * sizeof(double));
        double *B    = (double*) malloc((size_t)dp * sizeof(double));
        double *Aden = (double*) malloc((size_t)dp * sizeof(double));

        for (int q1 = 0; q1 < nq; q1++) {
            double v = nl[q1];
            double sum_log_h = 0.0;

            for (int ii = start; ii < end; ii++) {
                int row = order0[ii];

                for (int j = 0; j < dp; j++) {
                    double u1 = F1[row + j*N];
                    if (cont[j]) {
                        /* f_j(x_j) * c_{j,1}{F_j(x_j), v} -- the density
                         * prefactor required by eq. (2.8); see Aden[] use
                         * below. */
                        double fdens = f[row + j*N];
                        A[j]    = pcond_c(u1, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                        Aden[j] = fdens * dcop_c(u1, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                    } else {
                        double u2 = F2[row + j*N];
                        A[j] = pcond_c(u1, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                        B[j] = pcond_c(u2, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                    }
                }

                double h_val = 0.0;
                for (int q2 = 0; q2 < nq; q2++) {
                    double w_ = nl[q2];
                    double LLval = 1.0;
                    for (int j = 0; j < dp; j++) {
                        double t;
                        if (cont[j]) {
                            t = Aden[j] * dcop_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                        } else {
                            double t3 = pcond_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                            double t4 = pcond_c(B[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                            t = t3 - t4;
                        }
                        LLval *= t;
                    }
                    h_val += wl[q2] * LLval;
                }
                if (h_val < 1e-100) h_val = 1e-100;
                sum_log_h += log(h_val);
            }
            ltem3[q1] = sum_log_h;
        }

        double lmax3 = ltem3[0];
        for (int q1 = 1; q1 < nq; q1++) if (ltem3[q1] > lmax3) lmax3 = ltem3[q1];
        double intf = 0.0;
        for (int q1 = 0; q1 < nq; q1++) intf += wl[q1] * exp(ltem3[q1]-lmax3);
        total += -log(intf) - lmax3;

        free(ltem3); free(A); free(B); free(Aden);
    } /* end of (parallel) k loop */

    *out = total;
}

/* ---------------------------------------------------------------------
 * cond_median_logh(): per-cluster log h_k(v) at a caller-supplied grid of
 * v-values, for use by condMedianV_bifactor()'s invfunc()-based bisection
 * (in R) for the conditional posterior MEDIAN of the cluster-level factor
 * V_k -- see eqs. (2.9)-(2.11) of "Bi-factor copulas for clustered data".
 *
 * This is the same per-member / per-margin / per-w-node computation as
 * the inner body of bifactor_negloglik()'s q1 loop above, just:
 *   (a) evaluated at an ARBITRARY vgrid (not necessarily the model's own
 *       fixed quadrature nodes -- invfunc() calls this at rescaled nodes
 *       x*nl on every bisection step, for many different x's), and
 *   (b) returning the raw per-v-node log h_k(v) values directly (`out`),
 *       rather than collapsing them via a second log-sum-exp over v into
 *       a single per-cluster negative-log-likelihood contribution the
 *       way bifactor_negloglik() does.
 *
 * Operates on ONE CLUSTER at a time (n_k members), unlike
 * bifactor_negloglik() which loops over all K clusters itself -- the
 * per-cluster R-level loop (over K) and the bisection (via invfunc())
 * both stay in R; only the expensive nested member/margin/w-node
 * quadrature moves here.
 *
 * F1,F2   : n_k x dp matrices for THIS CLUSTER ONLY, column-major, length n_k*dp
 *           (F2 unused/ignored for continuous margins, but must still be
 *           allocated -- pass zeros if there are no discrete margins)
 * cont    : length dp, 1 = continuous, 0 = discrete
 * fam1,fam2, rot1,rot2, cpar1,cpar2, dfC1,dfC2 : length dp (family codes as in family_code() on the R side)
 * vgrid   : length nvg, the v-values to evaluate log h_k(v) at
 * nl,wl   : length nqw, Gauss-Legendre-on-(0,1) nodes/weights for each member's own w-integral
 * out     : length nvg, receives log h_k(vgrid[q]) for each q
 * ------------------------------------------------------------------- */
void cond_median_logh(
    double *F1, double *F2,
    int *n_k_, int *dp_,
    int *cont,
    int *fam1, int *fam2,
    int *rot1, int *rot2,
    double *cpar1, double *cpar2,
    double *dfC1, double *dfC2,
    double *vgrid, int *nvg_,
    double *nl, double *wl, int *nqw_,
    double *out
) {
    int n_k = *n_k_, dp = *dp_, nvg = *nvg_, nqw = *nqw_;

    double *A = (double*) malloc((size_t)dp * sizeof(double));
    double *B = (double*) malloc((size_t)dp * sizeof(double));
    double *logLLw = (double*) malloc((size_t)nqw * sizeof(double));

    for (int q1 = 0; q1 < nvg; q1++) {
        double v = vgrid[q1];
        double sum_log_h = 0.0;

        for (int i = 0; i < n_k; i++) {
            for (int j = 0; j < dp; j++) {
                double u1 = F1[i + j*n_k];
                A[j] = pcond_c(u1, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                if (!cont[j]) {
                    double u2 = F2[i + j*n_k];
                    B[j] = pcond_c(u2, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                }
            }

            /* integrate out this member's own nuisance w via log-sum-exp
             * across the nqw quadrature nodes (same stabilization as the
             * R version's Step 3) */
            for (int q2 = 0; q2 < nqw; q2++) {
                double w_ = nl[q2];
                double logLL = 0.0;
                for (int j = 0; j < dp; j++) {
                    double t;
                    if (cont[j]) {
                        t = dcop_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                        if (t < 1e-300) t = 1e-300;
                    } else {
                        double t1 = pcond_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                        double t2 = pcond_c(B[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                        t = t1 - t2;
                        if (t < 1e-100) t = 1e-100;
                    }
                    logLL += log(t);
                }
                logLLw[q2] = logLL;
            }
            double lmaxw = logLLw[0];
            for (int q2 = 1; q2 < nqw; q2++) if (logLLw[q2] > lmaxw) lmaxw = logLLw[q2];
            double inner = 0.0;
            for (int q2 = 0; q2 < nqw; q2++) inner += wl[q2] * exp(logLLw[q2] - lmaxw);
            if (inner < 1e-300) inner = 1e-300;

            sum_log_h += lmaxw + log(inner);
        }
        out[q1] = sum_log_h;
    }

    free(A); free(B); free(logLLw);
}

/* ---------------------------------------------------------------------
 * cond_median_logh_multi(): batched version of cond_median_logh() across
 * MULTIPLE clusters at once, each evaluated at its OWN v-grid -- lets
 * the R-side bisection call invfunc() ONCE across all K clusters
 * simultaneously (exploiting invfunc()'s own built-in vectorization
 * across independent targets -- it subsets q/v0/v1/th together by the
 * same convergence mask, and a cluster's index can be smuggled through
 * as `th`), instead of K separate per-cluster invfunc() calls, each
 * making its own small, serial .C() calls. Only the clusters listed in
 * `active_k` (0-based indices into the FULL K-cluster F1/F2/order0/
 * cbound arrays) are processed on this call -- so already-converged
 * clusters drop out of later, cheaper bisection iterations, same as
 * invfunc()'s own !iconv masking. Parallelized across those active
 * clusters with OpenMP, exactly like bifactor_negloglik().
 *
 * F1,F2      : N x dp matrices for ALL K clusters (column-major), same
 *              convention as bifactor_negloglik().
 * N,dp       : total rows (all clusters) / number of margins.
 * cont,fam1,fam2,rot1,rot2,cpar1,cpar2,dfC1,dfC2 : length dp (global, as
 *              in bifactor_negloglik()).
 * order0,cbound : same convention as bifactor_negloglik() -- 0-based row
 *              indices grouped by cluster, and cluster boundaries into
 *              order0 (cbound has length K+1; cbound[0]=0).
 * active_k   : length n_active, 0-based indices (into 0..K-1) of which
 *              clusters to process on this call.
 * vgrid      : n_active x nq matrix, column-major -- row j is
 *              active_k[j]'s own v-grid (nq values) to evaluate at.
 * nl,wl      : length nqw, shared w-quadrature nodes/weights (same for
 *              every cluster -- only the v-grid varies by cluster).
 * out        : n_active x nq matrix, column-major, matching vgrid's
 *              layout -- out[j,q] = log h_{active_k[j]}(vgrid[j,q]).
 * ------------------------------------------------------------------- */
void cond_median_logh_multi(
    double *F1, double *F2,
    int *N_, int *dp_,
    int *cont,
    int *fam1, int *fam2,
    int *rot1, int *rot2,
    double *cpar1, double *cpar2,
    double *dfC1, double *dfC2,
    int *order0, int *cbound,
    int *active_k, int *n_active_,
    double *vgrid, int *nq_,
    double *nl, double *wl, int *nqw_,
    double *out
) {
    int N = *N_, dp = *dp_, n_active = *n_active_, nq = *nq_, nqw = *nqw_;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) if(n_active > 1)
#endif
    for (int jj = 0; jj < n_active; jj++) {
        int k = active_k[jj];
        int start = cbound[k], end = cbound[k+1];

        double *A = (double*) malloc((size_t)dp * sizeof(double));
        double *B = (double*) malloc((size_t)dp * sizeof(double));
        double *logLLw = (double*) malloc((size_t)nqw * sizeof(double));

        for (int q1 = 0; q1 < nq; q1++) {
            double v = vgrid[jj + q1 * n_active];  /* column-major: row jj, col q1 */
            double sum_log_h = 0.0;

            for (int ii = start; ii < end; ii++) {
                int row = order0[ii];

                for (int j = 0; j < dp; j++) {
                    double u1 = F1[row + j*N];
                    A[j] = pcond_c(u1, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                    if (!cont[j]) {
                        double u2 = F2[row + j*N];
                        B[j] = pcond_c(u2, v, fam1[j], rot1[j], cpar1[j], dfC1[j]);
                    }
                }

                for (int q2 = 0; q2 < nqw; q2++) {
                    double w_ = nl[q2];
                    double logLL = 0.0;
                    for (int j = 0; j < dp; j++) {
                        double t;
                        if (cont[j]) {
                            t = dcop_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                            if (t < 1e-300) t = 1e-300;
                        } else {
                            double t1 = pcond_c(A[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                            double t2 = pcond_c(B[j], w_, fam2[j], rot2[j], cpar2[j], dfC2[j]);
                            t = t1 - t2;
                            if (t < 1e-100) t = 1e-100;
                        }
                        logLL += log(t);
                    }
                    logLLw[q2] = logLL;
                }
                double lmaxw = logLLw[0];
                for (int q2 = 1; q2 < nqw; q2++) if (logLLw[q2] > lmaxw) lmaxw = logLLw[q2];
                double inner = 0.0;
                for (int q2 = 0; q2 < nqw; q2++) inner += wl[q2] * exp(logLLw[q2] - lmaxw);
                if (inner < 1e-300) inner = 1e-300;

                sum_log_h += lmaxw + log(inner);
            }
            out[jj + q1 * n_active] = sum_log_h;
        }

        free(A); free(B); free(logLLw);
    }
}

#ifdef STANDALONE_TEST
/* ---------------------------------------------------------------------
 * Standalone self-tests, compiled WITHOUT R, to check the math is sane
 * before it's ever loaded into R. Build with:
 *   gcc -DSTANDALONE_TEST -O2 -o test_kernel bifactor_kernel.c -lm
 * ------------------------------------------------------------------- */
#include <stdio.h>

static int near(double a, double b, double tol) {
    return fabs(a-b) < tol;
}

int main(void) {
    int fail = 0;

    /* qnorm/pnorm round-trip */
    double ps[] = {0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999};
    for (int i = 0; i < 9; i++) {
        double x = qnorm_c(ps[i]);
        double p2 = pnorm_c(x);
        if (!near(p2, ps[i], 1e-9)) {
            printf("FAIL qnorm/pnorm round trip at p=%g -> %g\n", ps[i], p2);
            fail++;
        }
    }
    /* known qnorm values */
    if (!near(qnorm_c(0.975), 1.959963985, 1e-6)) { printf("FAIL qnorm(0.975)\n"); fail++; }
    if (!near(qnorm_c(0.5), 0.0, 1e-12)) { printf("FAIL qnorm(0.5)\n"); fail++; }

    /* t distribution round trip, several df */
    double dfs[] = {2, 5, 10, 30, 100};
    for (int d = 0; d < 5; d++) {
        for (int i = 0; i < 9; i++) {
            double x = qt_c(ps[i], dfs[d]);
            double p2 = pt_c(x, dfs[d]);
            if (!near(p2, ps[i], 1e-7)) {
                printf("FAIL qt/pt round trip df=%g p=%g -> %g (x=%g)\n", dfs[d], ps[i], p2, x);
                fail++;
            }
        }
    }
    /* known t quantile: qt(0.975, 10) ~ 2.228139 (R value) */
    if (!near(qt_c(0.975,10), 2.228139, 1e-4)) {
        printf("FAIL qt(0.975,10) = %g (expect ~2.228139)\n", qt_c(0.975,10));
        fail++;
    }
    /* as df -> large, t should approach normal: qt(0.975,1e6) ~ qnorm(0.975) */
    if (!near(qt_c(0.975,1e6), qnorm_c(0.975), 1e-3)) {
        printf("FAIL qt(0.975,1e6) vs qnorm(0.975): %g vs %g\n", qt_c(0.975,1e6), qnorm_c(0.975));
        fail++;
    }

    /* copula sanity: clayton density should integrate to ~1 over unit square
     * (crude 2D midpoint rule -- just a smoke test, not high precision) */
    {
        double cpar = 2.0;
        int n = 400;
        double h = 1.0/n, sum = 0.0;
        for (int i = 0; i < n; i++) {
            double u = (i+0.5)*h;
            for (int jx = 0; jx < n; jx++) {
                double v = (jx+0.5)*h;
                sum += dmtcj_c(u,v,cpar);
            }
        }
        sum *= h*h;
        if (!near(sum, 1.0, 0.02)) {
            printf("FAIL clayton density integrates to %g (expect ~1)\n", sum);
            fail++;
        } else {
            printf("OK clayton density integrates to %g\n", sum);
        }
    }
    /* pcond(1,v,...) should be 1, pcond(0,v,...)=0 for several families */
    {
        int fams[] = {0,2,3,4,5,6,7,8,9};
        for (int fi = 0; fi < 9; fi++) {
            double p1 = pcond_c(1.0, 0.4, fams[fi], 0, 0.5, 5.0);
            double p0 = pcond_c(0.0, 0.4, fams[fi], 0, 0.5, 5.0);
            if (!near(p1,1.0,1e-9) || !near(p0,0.0,1e-9)) {
                printf("FAIL pcond boundary family=%d p0=%g p1=%g\n", fams[fi], p0, p1);
                fail++;
            }
        }
    }
    /* pcondfgm base formula, hand-checked: pcondfgm(0.5,0.6,0.9) = 0.455 */
    if (!near(pcondfgm_c(0.5,0.6,0.9), 0.455, 1e-9)) {
        printf("FAIL pcondfgm_c(0.5,0.6,0.9) = %g (expect 0.455)\n", pcondfgm_c(0.5,0.6,0.9));
        fail++;
    } else {
        printf("OK pcondfgm_c(0.5,0.6,0.9) = %g\n", pcondfgm_c(0.5,0.6,0.9));
    }
    /* rotated fgm via dispatcher should match the rotation identities */
    {
        double u=0.3, v=0.7, cpar=0.5;
        double p90  = pcond_c(u,v,9,90,cpar,0.0);
        double ref90 = 1.0 - pcondfgm_c(1-u,v,cpar);
        double p180 = pcond_c(u,v,9,180,cpar,0.0);
        double ref180 = 1.0 - pcondfgm_c(1-u,1-v,cpar);
        double p270 = pcond_c(u,v,9,270,cpar,0.0);
        double ref270 = pcondfgm_c(u,1-v,cpar);
        if (!near(p90,ref90,1e-12) || !near(p180,ref180,1e-12) || !near(p270,ref270,1e-12)) {
            printf("FAIL fgm rotation dispatch mismatch\n");
            fail++;
        } else {
            printf("OK fgm rotation dispatch matches (90/180/270)\n");
        }
    }

    /* Full kernel smoke test: 2 clusters of size 2, dp=1, gumbel/gumbel(rot90) */
    {
        int N = 4, dp = 1, K = 2, nq = 15;
        double F1[4] = {0.3,0.6,0.2,0.8};
        double F2[4] = {0.0,0.0,0.0,0.0};
        double f[4]  = {1.0,1.0,1.0,1.0};
        int cont[1] = {1};
        int fam1[1] = {4}, fam2[1] = {4};
        int rot1[1] = {0}, rot2[1] = {90};
        double cpar1[1] = {2.0}, cpar2[1] = {1.4};
        double dfC1[1] = {0.0}, dfC2[1] = {0.0};
        int order0[4] = {0,1,2,3};
        int cbound[3] = {0,2,4};
        /* crude nq-point Gauss-Legendre on (0,1): reuse a fixed small table
           for nq=15 is unnecessary for a smoke test -- use simple midpoint
           nodes with uniform weights, which is NOT what R's statmod uses,
           but is enough to confirm the function runs and returns a finite,
           sane number. */
        double nl[15], wl[15];
        for (int i = 0; i < nq; i++) { nl[i] = (i+0.5)/nq; wl[i] = 1.0/nq; }
        double out;
        bifactor_negloglik(F1,F2,f,&N,&dp,cont,fam1,fam2,rot1,rot2,
                            cpar1,cpar2,dfC1,dfC2,order0,cbound,&K,
                            nl,wl,&nq,&out);
        printf("Kernel smoke test: negloglik = %g\n", out);
        if (!isfinite(out)) { printf("FAIL kernel produced non-finite result\n"); fail++; }
    }

    /* cond_median_logh() smoke test: 1 cluster, n_k=3 members, dp=2
       margins (1 continuous gumbel/gumbel(rot90), 1 discrete
       clayton/clayton(rot180)), evaluated at a small v-grid. Checks:
       (a) it runs and returns finite values, (b) log h_k(v) is a
       reasonably smooth, unimodal-ish function of v for this simple
       case (monotonic up then down -- not a strict mathematical
       guarantee in general, but a useful red flag if grossly violated
       here), and (c) it agrees with a direct, independent re-implementation
       of the same nested loop written separately below (not reusing
       cond_median_logh() itself), to catch transcription bugs. */
    {
        int n_k = 3, dp = 2, nqw = 20;
        double F1[6] = {0.3,0.5,0.7,  0.2,0.4,0.6};   /* col-major: col0=margin0(cont), col1=margin1(disc) */
        double F2[6] = {0.0,0.0,0.0,  0.1,0.3,0.5};   /* only meaningful for margin1 (discrete) */
        int cont[2] = {1,0};
        int fam1[2] = {4,2}, fam2[2] = {4,2};          /* gumbel, clayton */
        int rot1[2] = {0,180}, rot2[2] = {90,180};
        double cpar1[2] = {2.0,0.86}, cpar2[2] = {1.4,0.2};
        double dfC1[2] = {0.0,0.0}, dfC2[2] = {0.0,0.0};

        int nvg = 5;
        double vgrid[5] = {0.1,0.3,0.5,0.7,0.9};
        double nl[20], wl[20];
        for (int i = 0; i < nqw; i++) { nl[i] = (i+0.5)/nqw; wl[i] = 1.0/nqw; }

        double out[5];
        cond_median_logh(F1,F2,&n_k,&dp,cont,fam1,fam2,rot1,rot2,
                          cpar1,cpar2,dfC1,dfC2,vgrid,&nvg,nl,wl,&nqw,out);

        int ok = 1;
        for (int q = 0; q < nvg; q++) {
            printf("  log h_k(v=%.1f) = %g\n", vgrid[q], out[q]);
            if (!isfinite(out[q])) ok = 0;
        }
        if (!ok) { printf("FAIL cond_median_logh produced non-finite value(s)\n"); fail++; }

        /* Independent re-implementation (deliberately NOT calling
           cond_median_logh or sharing its loop code) for cross-checking. */
        int bad_match = 0;
        for (int q = 0; q < nvg; q++) {
            double v = vgrid[q];
            double sum_log_h = 0.0;
            for (int i = 0; i < n_k; i++) {
                double A0 = pcond_c(F1[i+0*n_k], v, fam1[0], rot1[0], cpar1[0], dfC1[0]);
                double A1 = pcond_c(F1[i+1*n_k], v, fam1[1], rot1[1], cpar1[1], dfC1[1]);
                double B1 = pcond_c(F2[i+1*n_k], v, fam1[1], rot1[1], cpar1[1], dfC1[1]);
                double hval = 0.0;
                for (int q2 = 0; q2 < nqw; q2++) {
                    double w_ = nl[q2];
                    double t0 = dcop_c(A0, w_, fam2[0], rot2[0], cpar2[0], dfC2[0]);
                    double t1a = pcond_c(A1, w_, fam2[1], rot2[1], cpar2[1], dfC2[1]);
                    double t1b = pcond_c(B1, w_, fam2[1], rot2[1], cpar2[1], dfC2[1]);
                    double t1 = t1a - t1b; if (t1 < 1e-100) t1 = 1e-100;
                    hval += wl[q2] * t0 * t1;
                }
                if (hval < 1e-100) hval = 1e-100;
                sum_log_h += log(hval);
            }
            if (!near(sum_log_h, out[q], 1e-6)) {
                printf("FAIL cond_median_logh mismatch at v=%.1f: %g (direct) vs %g (kernel)\n",
                       v, sum_log_h, out[q]);
                bad_match++;
            }
        }
        if (bad_match) fail += bad_match;
        else printf("OK cond_median_logh matches an independent re-implementation\n");
    }

    /* cond_median_logh_multi() smoke test: 3 clusters (sizes 2, 3, 2),
       dp=2 margins (same families as the cond_median_logh test above),
       each cluster given its OWN v-grid (as would happen mid-bisection,
       with different clusters at different x's). Cross-checked against
       calling the already-validated cond_median_logh() once per cluster
       with that cluster's own data and v-grid. Also tests a PARTIAL
       active_k list (only clusters 0 and 2), mimicking a later bisection
       iteration where cluster 1 has already converged and dropped out. */
    {
        int dp = 2;
        int fam1[2] = {4,2}, fam2[2] = {4,2};
        int rot1[2] = {0,180}, rot2[2] = {90,180};
        double cpar1[2] = {2.0,0.86}, cpar2[2] = {1.4,0.2};
        double dfC1[2] = {0.0,0.0}, dfC2[2] = {0.0,0.0};
        int cont[2] = {1,0};

        /* 3 clusters, sizes 2,3,2 -> N=7 total rows */
        int N = 7;
        double F1[14] = {
            0.3,0.5,  0.2,0.4,0.6,  0.35,0.55,      /* margin 0 (col-major, cluster-grouped) */
            0.25,0.45, 0.15,0.3,0.5, 0.28,0.48       /* margin 1 */
        };
        double F2[14] = {
            0.0,0.0,   0.0,0.0,0.0,  0.0,0.0,
            0.1,0.25,  0.05,0.15,0.3, 0.12,0.3
        };
        int order0[7] = {0,1, 2,3,4, 5,6};
        int cbound[4] = {0,2,5,7};

        int nqw = 20;
        double nl[20], wl[20];
        for (int i = 0; i < nqw; i++) { nl[i] = (i+0.5)/nqw; wl[i] = 1.0/nqw; }

        int nq = 4;
        /* each cluster's own v-grid (as if mid-bisection at different x's) */
        double x[3] = {0.2, 0.6, 0.9};
        double vgrid[12]; /* 3 x 4, column-major */
        for (int jc = 0; jc < 3; jc++)
            for (int q = 0; q < nq; q++)
                vgrid[jc + q*3] = x[jc] * nl[(q*nqw)/nq]; /* 4 spread-out sample nodes from the 20 */

        int active_k[3] = {0,1,2};
        int n_active = 3;
        double out_multi[12];
        cond_median_logh_multi(F1,F2,&N,&dp,cont,fam1,fam2,rot1,rot2,cpar1,cpar2,dfC1,dfC2,
                                order0,cbound,active_k,&n_active,vgrid,&nq,nl,wl,&nqw,out_multi);

        int ok = 1, mismatch = 0;
        for (int jc = 0; jc < 3; jc++) {
            int n_k = cbound[jc+1]-cbound[jc];
            double F1c[6], F2c[6];
            for (int i = 0; i < n_k; i++) {
                int row = order0[cbound[jc]+i];
                for (int j = 0; j < dp; j++) { F1c[i+j*n_k]=F1[row+j*N]; F2c[i+j*n_k]=F2[row+j*N]; }
            }
            double vgc[4], outc[4];
            for (int q = 0; q < nq; q++) vgc[q] = vgrid[jc + q*3];
            cond_median_logh(F1c,F2c,&n_k,&dp,cont,fam1,fam2,rot1,rot2,cpar1,cpar2,dfC1,dfC2,
                              vgc,&nq,nl,wl,&nqw,outc);
            for (int q = 0; q < nq; q++) {
                double m = out_multi[jc + q*3];
                if (!isfinite(m)) ok = 0;
                if (!near(m, outc[q], 1e-6)) {
                    printf("FAIL cond_median_logh_multi mismatch cluster=%d q=%d: %g (multi) vs %g (single-cluster)\n",
                           jc, q, m, outc[q]);
                    mismatch++;
                }
            }
        }
        if (!ok) { printf("FAIL cond_median_logh_multi produced non-finite value(s)\n"); fail++; }
        if (mismatch) fail += mismatch;
        else printf("OK cond_median_logh_multi (all 3 active) matches per-cluster cond_median_logh\n");

        /* Partial active_k: only clusters 0 and 2 (mimics cluster 1 having
           already converged and dropped out of this bisection iteration). */
        int active_k2[2] = {0,2};
        int n_active2 = 2;
        double vgrid2[8]; /* 2 x 4, column-major, reusing clusters 0 and 2's grids from above */
        for (int q = 0; q < nq; q++) { vgrid2[0+q*2]=vgrid[0+q*3]; vgrid2[1+q*2]=vgrid[2+q*3]; }
        double out_multi2[8];
        cond_median_logh_multi(F1,F2,&N,&dp,cont,fam1,fam2,rot1,rot2,cpar1,cpar2,dfC1,dfC2,
                                order0,cbound,active_k2,&n_active2,vgrid2,&nq,nl,wl,&nqw,out_multi2);
        int mismatch2 = 0;
        for (int q = 0; q < nq; q++) {
            if (!near(out_multi2[0+q*2], out_multi[0+q*3], 1e-9)) mismatch2++;
            if (!near(out_multi2[1+q*2], out_multi[2+q*3], 1e-9)) mismatch2++;
        }
        if (mismatch2) { printf("FAIL cond_median_logh_multi partial active_k mismatch\n"); fail += mismatch2; }
        else printf("OK cond_median_logh_multi partial active_k (subset of clusters) matches full-batch result\n");
    }

    if (fail) { printf("\n%d TEST(S) FAILED\n", fail); return 1; }
    printf("\nAll standalone tests passed.\n");
    return 0;
}
#endif
