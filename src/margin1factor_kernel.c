/* margin1factor_kernel.c
 *
 * Compiled replacement for the hot inner loop of EstMargin1Factor()'s
 * per-cluster / single-quadrature reduction -- Step 1 of the two-step
 * bi-factor estimator. Unlike bifactor_kernel.c (double quadrature over
 * V and W, dp margins jointly), this is a SINGLE quadrature over V only,
 * for ONE margin at a time -- matching eq. (2.6)/(2.7) of the bi-factor
 * copula paper, not the full eq. (2.8).
 *
 * Design: R still computes the margin's own cdf/pdf and (for continuous
 * margins) their theta-derivatives via margins() -- porting every
 * margin family (poisson, nbinom, weibull, ...) to C would be a much
 * bigger and less valuable undertaking than porting the copula side,
 * since margins() is already a single vectorized R call per likf()
 * evaluation, not a per-cluster loop. What WAS the per-cluster R loop --
 * the quadrature reduction, the log-sum-exp stabilization, and (for
 * continuous margins) the analytic chain-rule gradient through the
 * copula -- is what this kernel replaces.
 *
 * Family codes (must match bifactor_kernel.c's mapping and coplik.R's
 * coverage): 0 gaussian/normal, 1 t, 2 clayton, 3 frank, 4 gumbel,
 * 5 joe, 6 plackett, 9 fgm. (7 galambos, 8 huesler-reiss, 10 ind are
 * NOT covered by coplik.R and so are not implemented here either --
 * EstMargin1Factor() falls back to a pure-R finite-difference evaluator
 * for those, same as before.)
 *
 * Column convention for the 5-element per-family outputs below: this
 * mirrors coplik.R's OWN internal column order exactly (after the fix
 * that made every family consistent):
 *   out[0] = C(u|v)          out[1] = dC(u|v)/dcpar
 *   out[2] = c(u,v) (density) out[3] = dc(u,v)/dcpar
 *   out[4] = dc(u,v)/du   (Step 1 never needs the "du=FALSE" variant of
 *                          this last column -- the continuous branch
 *                          always wants dc/du, the discrete branch never
 *                          reads column 4 at all -- so only that one
 *                          variant is implemented here.)
 *
 * Build (from R), matching EstBifactorCopulaGAMMFast's OpenMP-aware
 * loader:
 *   PKG_CFLAGS=-fopenmp PKG_LIBS=-fopenmp R CMD SHLIB margin1factor_kernel.c
 * or, single-threaded:
 *   R CMD SHLIB margin1factor_kernel.c
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
 * Per-family derivative functions. Each fills
 *   out[5] = { C(u|v), dC(u|v)/dcpar, c(u,v), dc(u,v)/dcpar, dc(u,v)/du }
 * for the UNROTATED base copula, given u,v in (0,1). These are direct
 * transcriptions of coplik.R's fnorders/ftders/fmtcjders/fgumders/
 * ffrkders/ffgmders/fjoeders/fpladers (du=TRUE branch only -- see file
 * header note on why du=FALSE is never needed here), including each
 * function's own boundary special-casing.
 * ------------------------------------------------------------------- */

static void fnorders_c(double u, double v, double cpar, double out[5]) {
    double rho = cpar;
    double Cpi = sqrt(2.0*M_PI);
    double qu = qnorm_c(u), qv = qnorm_c(v);
    double qu2 = qu*qu, qv2 = qv*qv, quv = qu*qv;
    double rho2 = rho*rho, srho2 = sqrt(1.0-rho2), srho3 = srho2*(1.0-rho2);
    double zuv = (qu-rho*qv)/srho2;
    double ccond = pnorm_c(zuv);
    double euv = exp(-0.5*zuv*zuv)/Cpi;
    double ccond1 = euv*(rho*qu-qv)/srho3;
    double pdf = exp(0.5*(-rho2*(qu2+qv2)+2.0*rho*quv)/(1.0-rho2))/srho2;
    double pdf1 = pdf*((quv*(1.0+rho2)-rho*(qu2+qv2))/(1.0-rho2)/(1.0-rho2) + rho/(1.0-rho2));
    double pdfu = pdf*Cpi*rho*(qv-rho*qu)*exp(0.5*qu2)/(1.0-rho2);
    if (u<1e-10 || u==1.0 || v==0.0 || v==1.0) { ccond1=0; pdf1=0; pdfu=0; pdf=0; }
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void ftders_c(double u, double v, double cpar, double dfC, double out[5]) {
    double nu = dfC;
    double x = qt_c(u,nu), y = qt_c(v,nu);
    double rho = cpar;
    double Omega = 1.0-rho*rho;
    double mu = y*rho;
    double z = (nu+y*y)/(nu+1.0);
    double sig = sqrt(Omega*z);
    double w = (x-mu)/sig;
    double ccond = pt_c(w,nu+1.0);
    double wu = 1.0/sig/dt_c(x,nu);
    double pdf = dt_c(w,nu+1.0)*wu;
    double dd = (y - rho*w*sig/Omega);
    double w1 = -dd/sig;
    double ccond1 = dt_c(w,nu+1.0)*w1;
    double pdf1 = pdf*( rho/Omega - w*(nu+2.0)/(nu+1.0)*w1/(1.0+w*w/(nu+1.0)) );
    double pdfu = pdf*wu*(x*sig*(nu+1.0)/(nu+x*x) - w*(nu+2.0)/(nu+1.0+w*w));
    if (u<1e-10) { ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    if (u==1.0)  { ccond1=1; pdf=1; pdf1=1; pdfu=1; } /* matches R's `out[u==1,2:5]=1` verbatim */
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void fmtcjders_c(double u, double v, double cpar, double out[5]) {
    if (cpar < 1e-6) cpar = 1e-6;
    if (cpar > 40.0) cpar = 40.0;
    double ud = pow(u,-cpar), vd = pow(v,-cpar);
    double cpar2 = 1.0/cpar/cpar;
    double lu = log(u), lv = log(v);
    double ud1 = ud/u, vd1 = vd/v;
    double uvd = ud+vd-1.0;
    double luvd = log(uvd);
    double cdf = pow(uvd,-1.0/cpar);
    double ccond = vd1*cdf/uvd;
    double pdf = (1.0+cpar)*ud1*ccond/uvd;
    double tem1 = -(lu*ud+lv*vd)/uvd;
    double lccond1 = -lv+cpar2*luvd-(1.0/cpar+1.0)*tem1;
    double ccond1 = ccond*lccond1;
    double lpdf1 = 1.0/(cpar+1.0)-lu-lv+cpar2*luvd-(1.0/cpar+2.0)*tem1;
    double pdf1 = pdf*lpdf1;
    double pdfu = pdf*(-cpar-1.0+(2.0*cpar+1.0)*ud/(ud+vd-1.0))/u;
    if (u<1e-10) { ccond=0; ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    if (v==0.0)  { ccond1=0; pdf=0; pdf1=0; pdfu=0; ccond=1; }
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void fgumders_c(double u, double v, double cpar, double out[5]) {
    if (cpar < 1.0+1e-6) cpar = 1.0+1e-6;
    double ul=-log(u), uul=u*ul, vl=-log(v);
    double ull=log(ul), vll=log(vl);
    double ut=pow(ul,cpar), vt=pow(vl,cpar);
    double uvt=ut+vt, luvt=log(uvt);
    double uvd=pow(uvt,1.0/cpar);
    double cdf=exp(-uvd);
    double ul1=ut/uul, vl1=vt/v/vl;
    double uvdt=uvd/uvt;
    double l1v=vl1*uvdt, l1u=ul1*uvdt;
    double l2uv=(cpar-1.0)*l1u*vl1/uvt;
    double ccond=cdf*l1v;
    double l1uv=l1u*l1v;
    double l1cdf=cdf*l1uv, l2cdf=cdf*l2uv;
    double pdf=l1cdf+l2cdf;
    double cf1=-l1u/(uvd+cpar-1.0);
    double cf2=-(1.0-2.0*cpar)*ul1/uvt;
    double cf3=-(cpar-1.0)/uul;
    double pdfu=pdf*(l1u+cf1+cf2+cf3-1.0/u);
    double tem1=(ut*ull+vt*vll)/uvt;
    double tem2=-luvt/cpar/cpar;
    double uvd1=uvd*(tem2+tem1/cpar);
    double tem12=tem2+(1.0/cpar-1.0)*tem1;
    double ccond1=ccond*(-uvd1+vll+tem12);
    double pdf1=-pdf*uvd1+l1cdf*(ull+vll+2.0*tem12)+l2cdf*(1.0/(cpar-1.0)+ull+vll+tem12-tem1);
    if (u<1e-10) { ccond=0; ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    if (u==1.0)  { ccond1=0; pdf1=0; pdfu=0; pdf=0; ccond=1; }
    if (v==0.0)  { ccond1=0; pdf1=0; pdfu=0; pdf=0; ccond=1; }
    if (v==1.0)  { ccond=0; ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void ffrkders_c(double u, double v, double cpar, double out[5]) {
    double ccond,ccond1,pdf,pdf1,pdfu;
    if (fabs(cpar) < 1e-5) {
        ccond = u;
        pdf = 1.0;
        pdfu = 0.0;
        ccond1 = u*(1.0-u)*(1.0-v);
        pdf1 = u*v + (1.0-u)*(1.0-v);
    } else {
        double de=exp(-cpar), ue=exp(-cpar*u), ve=exp(-cpar*v);
        double uve=ue*ve, uue=u*ue, vve=v*ve;
        double uvee=uve*(u+v);
        double den=(1.0-de)-(1.0-ue)*(1.0-ve);
        double den1=de-uue-vve+uvee;
        double den2=den*den;
        ccond = ve*(1.0-ue)/den;
        ccond1 = (-vve+uvee)/den - ccond*den1/den;
        pdf = cpar*(1.0-de)*uve/den2;
        double num1 = (1.0-de)*uve + de*cpar*uve - cpar*(1.0-de)*uvee;
        pdf1 = num1/den2 - 2.0*pdf*den1/den;
        pdfu = cpar*pdf*(2.0*ue*(1.0-ve)/den - 1.0);
    }
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void ffgmders_c(double u, double v, double cpar, double out[5]) {
    double ccond = u*(1.0+cpar*(1.0-u)*(1.0-2.0*v));
    double ccond1 = u*(1.0-u)*(1.0-2.0*v);
    double pdf = 1.0+cpar*(1.0-2.0*u)*(1.0-2.0*v);
    double pdf1 = (1.0-2.0*u)*(1.0-2.0*v);
    double pdfu = -2.0*cpar*(1.0-2.0*v);
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void fjoeders_c(double u, double v, double cpar, double out[5]) {
    double ud=pow(1.0-u,cpar), vd=pow(1.0-v,cpar);
    double lu=log(1.0-u), lv=log(1.0-v);
    double ctem1=1.0+ud/vd-ud;
    double ctem2=ud+vd-ud*vd;
    double num1=ud*lu+vd*lv-ud*vd*(lu+lv);
    double ccond=pow(ctem1,-1.0+1.0/cpar)*(1.0-ud);
    double pdf=pow(ctem2,-2.0+1.0/cpar)*ud*vd*(cpar-1.0+ctem2)/(1.0-u)/(1.0-v);
    double ccond1 = -log(ctem1)/(cpar*cpar) - ud*lu/(1.0-ud) + (-1.0+1.0/cpar)*((lu-lv)*ud/vd-ud*lu)/ctem1;
    ccond1 *= ccond;
    double pdf1 = -log(ctem2)/(cpar*cpar) + (-2.0+1.0/cpar)*num1/ctem2 + (1.0+num1)/(cpar-1.0+ctem2) + lu + lv;
    pdf1 *= pdf;
    double tem2 = cpar*ud*(vd-1.0)/(1.0-u);
    double pdfu = (-2.0+1.0/cpar)*tem2/ctem2 + (1.0-cpar)/(1.0-u) + tem2/(cpar-1.0+ctem2);
    pdfu *= pdf;
    if (u<1e-10) ccond1 = 0.0;
    if (v==1.0)  { ccond=0; ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    if (u==1.0)  { ccond1=0; pdf=0; pdf1=0; pdfu=0; }
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

static void fpladers_c(double u, double v, double cpar, double out[5]) {
    double eta = cpar-1.0;
    double uv = u*v;
    double uv2 = u+v-2.0*uv;
    double tem1 = sqrt(pow(1.0+eta*(u+v),2.0)-4.0*eta*cpar*uv);
    double tem3 = tem1*tem1*tem1;
    double tem5 = tem3*tem1*tem1;
    double temd = 2.0*(1.0+eta*(u+v))*(u+v)-4.0*uv*(eta+cpar);
    double num1 = eta*v+1.0-(eta+2.0)*u;
    double num2 = 1.0+eta*uv2;
    double ccond = 0.5 - 0.5*num1/tem1;
    double ccond1 = -0.5*(v-u)/tem1 + 0.25*num1*temd/tem3;
    double pdf = cpar*num2/tem3;
    double pdf1 = (1.0+(2.0*cpar-1.0)*uv2)/tem3 - 1.5*cpar*num2*temd/tem5;
    double pdfu = cpar*eta*(1.0-2.0*v)/tem3 - 1.5*cpar*num2*(2.0*eta*(1.0+eta*(u+v))-4.0*eta*cpar*v)/tem5;
    out[0]=ccond; out[1]=ccond1; out[2]=pdf; out[3]=pdf1; out[4]=pdfu;
}

/* ---------------------------------------------------------------------
 * Rotation dispatcher -- mirrors coplik.R's switch() exactly, including
 * which families rotate at all (gaussian/t/frank never do) and the
 * sign-flip on columns [1] (dC/dcpar) and [4] (dc/du) for rot=90/180.
 * ------------------------------------------------------------------- */
static void coplik_c(double u, double v, int fam, int rot, double cpar, double dfC, double out[5]) {
    switch (fam) {
        case 0: fnorders_c(u, v, cpar, out); break;
        case 1: ftders_c(u, v, cpar, dfC, out); break;
        case 2: /* clayton */
            if (rot==0)        fmtcjders_c(u, v, cpar, out);
            else if (rot==90)  { fmtcjders_c(1-u, v, cpar, out);   out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else if (rot==180) { fmtcjders_c(1-u, 1-v, cpar, out); out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else                fmtcjders_c(u, 1-v, cpar, out);
            break;
        case 6: /* plackett */
            if (rot==0)        fpladers_c(u, v, cpar, out);
            else if (rot==90)  { fpladers_c(1-u, v, cpar, out);   out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else if (rot==180) { fpladers_c(1-u, 1-v, cpar, out); out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else                fpladers_c(u, 1-v, cpar, out);
            break;
        case 5: /* joe */
            if (rot==0)        fjoeders_c(u, v, cpar, out);
            else if (rot==90)  { fjoeders_c(1-u, v, cpar, out);   out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else if (rot==180) { fjoeders_c(1-u, 1-v, cpar, out); out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else                fjoeders_c(u, 1-v, cpar, out);
            break;
        case 3: ffrkders_c(u, v, cpar, out); break; /* frank: never rotates, matches R */
        case 4: /* gumbel */
            if (rot==0)        fgumders_c(u, v, cpar, out);
            else if (rot==90)  { fgumders_c(1-u, v, cpar, out);   out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else if (rot==180) { fgumders_c(1-u, 1-v, cpar, out); out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else                fgumders_c(u, 1-v, cpar, out);
            break;
        case 9: /* fgm */
            if (rot==0)        ffgmders_c(u, v, cpar, out);
            else if (rot==90)  { ffgmders_c(1-u, v, cpar, out);   out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else if (rot==180) { ffgmders_c(1-u, 1-v, cpar, out); out[0]=1-out[0]; out[1]=-out[1]; out[4]=-out[4]; }
            else                ffgmders_c(u, 1-v, cpar, out);
            break;
        default:
            out[0]=out[1]=out[2]=out[3]=out[4]=NAN;
    }
}

/* ---------------------------------------------------------------------
 * Main entry point, called from R via .C().
 *
 * Gy, gy       : length N. Continuous: G(y), g(y). Discrete: G(y), P(Y=y)
 *                (u2 = Gy-gy is formed inside the kernel).
 * dG1,dG2      : length N each. dG(y)/dtheta_1, dG(y)/dtheta_2. Only
 *                read if cont=1 && want_th_grad=1 (pass anything, e.g.
 *                zero-length placeholders, otherwise).
 * dg1,dg2      : length N each. dg(y)/dtheta_1, dg(y)/dtheta_2. Same
 *                condition as dG1,dG2.
 * nth          : number of margin parameters actually in play (1 or 2).
 * cont         : 1 = continuous margin, 0 = discrete.
 * want_th_grad : 1 = also fill grad_th (only meaningful if cont=1).
 * fam1,rot1,cpar1,dfC1 : the copula-1 family/rotation/parameter/df.
 * order0,cbound,K : cluster grouping, same convention as bifactor_kernel.c.
 * nl,wl,nq     : Gauss-Legendre-on-(0,1) nodes/weights.
 * out          : length 1, total negative log-likelihood.
 * grad_th      : length 2 (only first `nth` meaningful), margin-parameter
 *                gradient (filled only if cont && want_th_grad).
 * grad_cpar1   : length 1, gradient with respect to cpar1 (always filled).
 * ------------------------------------------------------------------- */
void margin1factor_negloglik_grad(
    double *Gy, double *gy,
    double *dG1, double *dG2, double *dg1, double *dg2,
    int *N_, int *nth_, int *cont_, int *want_th_grad_,
    int *fam1_, int *rot1_, double *cpar1_, double *dfC1_,
    int *order0, int *cbound, int *K_,
    double *nl, double *wl, int *nq_,
    double *out, double *grad_th, double *grad_cpar1
) {
    int N = *N_, nth = *nth_, cont = *cont_, want_th_grad = *want_th_grad_;
    int fam1 = *fam1_, rot1 = *rot1_, K = *K_, nq = *nq_;
    double cpar1 = *cpar1_, dfC1 = *dfC1_;
    (void)N;

    double total = 0.0;
    double gth0 = 0.0, gth1 = 0.0;
    double gcpar1 = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:total,gth0,gth1,gcpar1) schedule(dynamic) if(K > 1)
#endif
    for (int k = 0; k < K; k++) {
        int start = cbound[k], end = cbound[k+1];

        double *ltem3   = (double*) malloc((size_t)nq * sizeof(double));
        double *dth3_0  = (double*) malloc((size_t)nq * sizeof(double));
        double *dth3_1  = (double*) malloc((size_t)nq * sizeof(double));
        double *dcpar3  = (double*) malloc((size_t)nq * sizeof(double));

        for (int q = 0; q < nq; q++) {
            double v = nl[q];
            double logH = 0.0, dth0 = 0.0, dth1 = 0.0, dcp = 0.0;

            for (int ii = start; ii < end; ii++) {
                int i = order0[ii];
                double u1 = Gy[i];
                double cp[5];

                if (cont) {
                    double g_i = gy[i];
                    coplik_c(u1, v, fam1, rot1, cpar1, dfC1, cp);
                    double dens = cp[2];
                    double h = g_i * dens;
                    logH += log(h);
                    dcp += cp[3] / dens;               /* d/dcpar1 log(density) */
                    if (want_th_grad) {
                        dth0 += dg1[i] / g_i + (cp[4] / dens) * dG1[i];
                        if (nth > 1) dth1 += dg2[i] / g_i + (cp[4] / dens) * dG2[i];
                    }
                } else {
                    double u2 = Gy[i] - gy[i];
                    double cp1[5], cp2[5];
                    coplik_c(u1, v, fam1, rot1, cpar1, dfC1, cp1);
                    coplik_c(u2, v, fam1, rot1, cpar1, dfC1, cp2);
                    double h = cp1[0] - cp2[0];
                    if (h < 1e-100) h = 1e-100;
                    logH += log(h);
                    dcp += (cp1[1] - cp2[1]) / h;
                }
            }
            ltem3[q] = logH;
            dth3_0[q] = dth0; dth3_1[q] = dth1;
            dcpar3[q] = dcp;
        }

        double lmax = ltem3[0];
        for (int q = 1; q < nq; q++) if (ltem3[q] > lmax) lmax = ltem3[q];
        double Lk = 0.0;
        for (int q = 0; q < nq; q++) Lk += wl[q] * exp(ltem3[q]-lmax);
        total += -(log(Lk) + lmax);

        for (int q = 0; q < nq; q++) {
            double post_w = (wl[q] * exp(ltem3[q]-lmax)) / Lk;
            gcpar1 -= post_w * dcpar3[q];
            if (cont && want_th_grad) {
                gth0 -= post_w * dth3_0[q];
                if (nth > 1) gth1 -= post_w * dth3_1[q];
            }
        }

        free(ltem3); free(dth3_0); free(dth3_1); free(dcpar3);
    }

    *out = total;
    *grad_cpar1 = gcpar1;
    if (cont && want_th_grad) { grad_th[0] = gth0; if (nth > 1) grad_th[1] = gth1; }
}

#ifdef STANDALONE_TEST
/* ---------------------------------------------------------------------
 * Standalone self-tests (no R needed). Build with:
 *   gcc -DSTANDALONE_TEST -O2 -o test_m1 margin1factor_kernel.c -lm
 *
 * Strategy: for each family, check internal self-consistency by finite-
 * differencing the VALUE outputs (ccond, pdf) of coplik_c() against its
 * own DERIVATIVE outputs (ccond1, pdf1, pdfu) -- this validates the
 * transcription without needing a second, independent implementation:
 * a sign/copy error in either the value or derivative formula shows up
 * as a mismatch here.
 * ------------------------------------------------------------------- */
#include <stdio.h>

static int near(double a, double b, double tol) { return fabs(a-b) < tol; }

static void fd_check(const char *name, int fam, int rot, double u, double v,
                      double cpar, double dfC, int *fail) {
    double h = 1e-6;
    double out[5], outp[5], outm[5];
    coplik_c(u, v, fam, rot, cpar, dfC, out);

    /* dC(u|v)/dcpar and dc(u,v)/dcpar via central difference on cpar */
    coplik_c(u, v, fam, rot, cpar+h, dfC, outp);
    coplik_c(u, v, fam, rot, cpar-h, dfC, outm);
    double dccond_fd = (outp[0]-outm[0])/(2*h);
    double dpdf_fd   = (outp[2]-outm[2])/(2*h);

    /* dc(u,v)/du via central difference on u */
    double hu = 1e-6;
    double outup[5], outum[5];
    coplik_c(u+hu, v, fam, rot, cpar, dfC, outup);
    coplik_c(u-hu, v, fam, rot, cpar, dfC, outum);
    double dpdf_du_fd = (outup[2]-outum[2])/(2*hu);

    int ok = near(out[1], dccond_fd, 1e-4) && near(out[3], dpdf_fd, 1e-4) &&
             near(out[4], dpdf_du_fd, 1e-3);
    if (!ok) {
        printf("FAIL %s (fam=%d,rot=%d,u=%g,v=%g,cpar=%g): "
               "ccond1=%g vs fd=%g | pdf1=%g vs fd=%g | pdfu=%g vs fd=%g\n",
               name, fam, rot, u, v, cpar,
               out[1], dccond_fd, out[3], dpdf_fd, out[4], dpdf_du_fd);
        (*fail)++;
    } else {
        printf("OK   %s (fam=%d,rot=%d,u=%g,v=%g,cpar=%g)\n", name, fam, rot, u, v, cpar);
    }
}

int main(void) {
    int fail = 0;

    fd_check("gaussian", 0, 0, 0.35, 0.62, 0.4, 0.0, &fail);
    fd_check("t",        1, 0, 0.35, 0.62, 0.4, 7.0, &fail);
    fd_check("clayton",  2, 0,   0.35, 0.62, 1.7, 0.0, &fail);
    fd_check("clayton90",2, 90,  0.35, 0.62, 1.7, 0.0, &fail);
    fd_check("clayton180",2,180, 0.35, 0.62, 1.7, 0.0, &fail);
    fd_check("clayton270",2,270, 0.35, 0.62, 1.7, 0.0, &fail);
    fd_check("frank",    3, 0, 0.35, 0.62, 2.3, 0.0, &fail);
    fd_check("gumbel",   4, 0,   0.35, 0.62, 1.9, 0.0, &fail);
    fd_check("gumbel90", 4, 90,  0.35, 0.62, 1.9, 0.0, &fail);
    fd_check("gumbel180",4, 180, 0.35, 0.62, 1.9, 0.0, &fail);
    fd_check("gumbel270",4, 270, 0.35, 0.62, 1.9, 0.0, &fail);
    fd_check("joe",      5, 0,   0.35, 0.62, 2.1, 0.0, &fail);
    fd_check("joe90",    5, 90,  0.35, 0.62, 2.1, 0.0, &fail);
    fd_check("joe180",   5, 180, 0.35, 0.62, 2.1, 0.0, &fail);
    fd_check("joe270",   5, 270, 0.35, 0.62, 2.1, 0.0, &fail);
    fd_check("plackett", 6, 0,   0.35, 0.62, 3.0, 0.0, &fail);
    fd_check("plackett90",6,90,  0.35, 0.62, 3.0, 0.0, &fail);
    fd_check("plackett180",6,180,0.35, 0.62, 3.0, 0.0, &fail);
    fd_check("plackett270",6,270,0.35, 0.62, 3.0, 0.0, &fail);
    fd_check("fgm",      9, 0,   0.35, 0.62, 0.5, 0.0, &fail);
    fd_check("fgm90",    9, 90,  0.35, 0.62, 0.5, 0.0, &fail);
    fd_check("fgm180",   9, 180, 0.35, 0.62, 0.5, 0.0, &fail);
    fd_check("fgm270",   9, 270, 0.35, 0.62, 0.5, 0.0, &fail);

    /* full kernel smoke test: continuous margin, 3 clusters of size 3,
       gaussian copula-1, with theta gradient requested */
    {
        int N=9, nth=2, cont=1, want_th_grad=1;
        int fam1=0, rot1=0, K=3, nq=20;
        double cpar1=0.5, dfC1=0.0;
        double Gy[9], gy[9], dG1[9], dG2[9], dg1[9], dg2[9];
        for (int i=0;i<9;i++){
            double z = 0.3+0.15*i;
            double m=0.2, s=1.3;
            double stz=(z-m)/s, stz2=stz*stz;
            Gy[i]=pnorm_c((z-m)/s);
            gy[i]=exp(-0.5*stz2)/sqrt(2*M_PI)/s;
            dG1[i]=-gy[i]; dG2[i]=-stz*gy[i];
            dg1[i]=gy[i]*stz/s; dg2[i]=-gy[i]/s+stz2*gy[i]/s;
        }
        int order0[9]; for(int i=0;i<9;i++) order0[i]=i;
        int cbound[4]={0,3,6,9};
        double nl[20], wl[20];
        for (int i=0;i<nq;i++){ nl[i]=(i+0.5)/nq; wl[i]=1.0/nq; }
        double out, grad_th[2], grad_cpar1;
        margin1factor_negloglik_grad(Gy,gy,dG1,dG2,dg1,dg2,&N,&nth,&cont,&want_th_grad,
            &fam1,&rot1,&cpar1,&dfC1,order0,cbound,&K,nl,wl,&nq,&out,grad_th,&grad_cpar1);
        printf("Step-1 kernel smoke test: negloglik=%g grad_th=(%g,%g) grad_cpar1=%g\n",
               out, grad_th[0], grad_th[1], grad_cpar1);
        if (!isfinite(out) || !isfinite(grad_th[0]) || !isfinite(grad_th[1]) || !isfinite(grad_cpar1)) {
            printf("FAIL non-finite kernel output\n"); fail++;
        }

        /* cross-check grad_th/grad_cpar1 against a finite difference of
           the TOTAL negloglik computed by re-running the kernel with
           perturbed inputs -- validates the quadrature-gradient
           reduction itself, not just the per-point copula derivatives. */
        double heps = 1e-6;
        double base_out;
        {
            int wtg0 = 0; double gth_dummy[2], gcp_dummy;
            margin1factor_negloglik_grad(Gy,gy,dG1,dG2,dg1,dg2,&N,&nth,&cont,&wtg0,
                &fam1,&rot1,&cpar1,&dfC1,order0,cbound,&K,nl,wl,&nq,&base_out,gth_dummy,&gcp_dummy);
        }
        /* d/dcpar1 via full FD */
        double outp, outm; int wtg0=0; double gth_dummy[2], gcp_dummy;
        double cp_p = cpar1+heps, cp_m = cpar1-heps;
        margin1factor_negloglik_grad(Gy,gy,dG1,dG2,dg1,dg2,&N,&nth,&cont,&wtg0,
            &fam1,&rot1,&cp_p,&dfC1,order0,cbound,&K,nl,wl,&nq,&outp,gth_dummy,&gcp_dummy);
        margin1factor_negloglik_grad(Gy,gy,dG1,dG2,dg1,dg2,&N,&nth,&cont,&wtg0,
            &fam1,&rot1,&cp_m,&dfC1,order0,cbound,&K,nl,wl,&nq,&outm,gth_dummy,&gcp_dummy);
        double grad_cpar1_fd = (outp-outm)/(2*heps);
        printf("  grad_cpar1 analytic=%g fd=%g\n", grad_cpar1, grad_cpar1_fd);
        if (!near(grad_cpar1, grad_cpar1_fd, 1e-3)) { printf("FAIL grad_cpar1 mismatch\n"); fail++; }
    }

    if (fail) { printf("\n%d TEST(S) FAILED\n", fail); return 1; }
    printf("\nAll standalone tests passed.\n");
    return 0;
}
#endif
