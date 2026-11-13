#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h> // for NULL

/* Declared `extern` here since each is a plain, non-static C function
 * defined in the corresponding .c file (not declared in any shared
 * header -- R only needs their addresses, not their full prototypes
 * elsewhere, and repeating the prototype here keeps this file
 * self-contained / independently readable). */
extern void bifactor_negloglik(
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
);

extern void cond_median_logh(
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
);

extern void cond_median_logh_multi(
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
);

extern void margin1factor_negloglik_grad(
    double *Gy, double *gy,
    double *dG1, double *dG2, double *dg1, double *dg2,
    int *N_, int *nth_, int *cont_, int *want_th_grad_,
    int *fam1_, int *rot1_, double *cpar1_, double *dfC1_,
    int *order0, int *cbound, int *K_,
    double *nl, double *wl, int *nq_,
    double *out, double *grad_th, double *grad_cpar1
);

static const R_CMethodDef CEntries[] = {
    {"bifactor_negloglik",           (DL_FUNC) &bifactor_negloglik,           21},
    {"cond_median_logh",             (DL_FUNC) &cond_median_logh,             19},
    {"cond_median_logh_multi",       (DL_FUNC) &cond_median_logh_multi,       23},
    {"margin1factor_negloglik_grad", (DL_FUNC) &margin1factor_negloglik_grad, 23},
    {NULL, NULL, 0}
};

void R_init_BiFactorCopulaGAMM(DllInfo *dll) {
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
