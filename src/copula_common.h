/* copula_common.h
 *
 * Standard normal and Student-t cdf/pdf/quantile helpers shared by
 * bifactor_kernel.c and margin1factor_kernel.c. Previously each file
 * carried its own identical `static` copy of these (so each compiled/
 * tested standalone without depending on the other); factored out here
 * so there's one copy to maintain. `static inline` keeps the same
 * per-translation-unit-private semantics the original `static` copies
 * had (no new exported symbols, no risk of a multiple-definition link
 * error), while letting the compiler inline them same as before.
 *
 * Include this BEFORE any code in either .c file that calls these
 * functions; both files still need <math.h> (for erf/erfc/lgamma/etc.)
 * included first, and M_PI/M_SQRT2 defined (both files already do this
 * themselves for portability, since some C libraries omit them).
 */
#ifndef COPULA_COMMON_H
#define COPULA_COMMON_H

/* ---------------------------------------------------------------------
 * Standard normal cdf/pdf and quantile function.
 * pnorm/dnorm are exact (erf/erfc based, C99 guarantees erf/erfc).
 * qnorm uses Acklam's rational approximation plus one Halley refinement
 * step (~1e-15 relative accuracy in practice).
 * ------------------------------------------------------------------- */
static inline double pnorm_c(double x) {
    return 0.5 * erfc(-x / M_SQRT2);
}
static inline double dnorm_c(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}
static inline double qnorm_c(double p) {
    static const double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const double p_low = 0.02425;
    double q, r, x;

    if (p <= 0.0) return -HUGE_VAL;
    if (p >= 1.0) return HUGE_VAL;

    if (p < p_low) {
        q = sqrt(-2.0 * log(p));
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= 1.0 - p_low) {
        q = p - 0.5;
        r = q*q;
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
              ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
    /* Halley refinement */
    double e = 0.5 * erfc(-x / M_SQRT2) - p;
    double u = e * sqrt(2.0 * M_PI) * exp(x * x / 2.0);
    x = x - u / (1.0 + x * u / 2.0);
    return x;
}

/* ---------------------------------------------------------------------
 * Student-t cdf/pdf/quantile via the regularized incomplete beta
 * function (classic continued-fraction evaluation).
 * ------------------------------------------------------------------- */
static inline double betacf_c(double a, double b, double x) {
    const int MAXIT = 200;
    const double EPS = 3.0e-12, FPMIN = 1.0e-300;
    int m, m2;
    double aa, c, d, del, h, qab, qam, qap;
    qab = a+b; qap = a+1.0; qam = a-1.0;
    c = 1.0; d = 1.0 - qab*x/qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0/d; h = d;
    for (m = 1; m <= MAXIT; m++) {
        m2 = 2*m;
        aa = m*(b-m)*x/((qam+m2)*(a+m2));
        d = 1.0+aa*d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0+aa/c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0/d;
        h *= d*c;
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
        d = 1.0+aa*d; if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0+aa/c; if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0/d;
        del = d*c;
        h *= del;
        if (fabs(del-1.0) < EPS) break;
    }
    return h;
}
static inline double betai_c(double a, double b, double x) {
    double bt;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    bt = exp(lgamma(a+b) - lgamma(a) - lgamma(b) + a*log(x) + b*log(1.0-x));
    if (x < (a+1.0)/(a+b+2.0))
        return bt * betacf_c(a,b,x) / a;
    else
        return 1.0 - bt * betacf_c(b,a,1.0-x) / b;
}
static inline double dt_c(double x, double df) {
    double logC = lgamma((df+1.0)/2.0) - lgamma(df/2.0) - 0.5*log(df*M_PI);
    return exp(logC - ((df+1.0)/2.0) * log(1.0 + x*x/df));
}
static inline double pt_c(double x, double df) {
    double xt = df/(df + x*x);
    double p = 0.5 * betai_c(df/2.0, 0.5, xt);
    if (x > 0.0) p = 1.0 - p;
    return p;
}
static inline double qt_c(double p, double df) {
    if (p <= 0.0) return -HUGE_VAL;
    if (p >= 1.0) return HUGE_VAL;
    /* Newton-Raphson on pt_c, started from the normal approximation,
     * with a bisection fallback bracket for robustness. */
    double lo = -1e5, hi = 1e5;
    double x = qnorm_c(p); /* good starting guess for moderate/large df */
    for (int it = 0; it < 50; it++) {
        double fx = pt_c(x, df) - p;
        if (fx > 0) hi = x; else lo = x;
        double dfx = dt_c(x, df);
        double step;
        if (dfx > 1e-300) {
            step = fx / dfx;
            double xn = x - step;
            if (xn <= lo || xn >= hi) {
                xn = 0.5*(lo+hi); /* bisection fallback */
            }
            if (fabs(xn - x) < 1e-12 * (1.0 + fabs(x))) { x = xn; break; }
            x = xn;
        } else {
            x = 0.5*(lo+hi);
        }
    }
    return x;
}

#endif /* COPULA_COMMON_H */
