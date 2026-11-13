#' condMedianV_bifactor.R
#'
#' Computes the conditional (posterior) MEDIAN of the cluster-level latent
#' factor V_k for each cluster, combining all d Y-margins and p X-margins
#' jointly, following eqs. (2.9)-(2.11) of "Bi-factor copulas for clustered
#' data": for cluster k with members i=1..n_k,
#'
#'   h_k(y_k,x_k,v) = prod_[i=1..n_k] h(y_ki, x_ki, v)                 (built from 2.9 via 2.5)
#'   h_k(y_k,x_k)   = integral_0^1 h_k(y_k,x_k,v) dv                     (2.10)
#'   posterior of V_k at v  =  h_k(y_k,x_k,v) / h_k(y_k,x_k)             (2.11)
#'
#' IMPORTANT (this is the correction from the previous version of this
#' file): `w` in (2.9) is a PER-OBSERVATION nuisance variable, not a
#' second cluster-level latent factor -- see section 2.2 of the paper,
#' where the auxiliary variables are written W_[k,i] (both a cluster index
#' k AND a member index i). It is integrated out separately for every
#' single cluster member i (to combine that member's own Y/X into one
#' joint density given V=v), then the n_k per-member results are
#' multiplied together, and ONLY THEN is the integral over v=V_k taken.
#' There is therefore only ONE latent variable with a meaningful
#' cluster-level posterior to report: V_k. This is exactly what
#' `bifactor_kernel.c`'s bifactor_negloglik() computes too (its q1 loop is
#' the outer integral over v; for each v, and EACH cluster member row
#' separately, it re-runs its own inner q2 loop over w and multiplies the
#' resulting per-member h_val's together via `sum_log_h`) -- this file
#' mirrors that same nested structure, but inverts the resulting posterior
#' CDF of V (via the same quadrature + invfunc-bisection pattern
#' MAP.continuous()/MAP.discrete() use) instead of returning a
#' log-likelihood.
#'
#' ASSUMPTIONS (please double-check against your fitted object -- I
#' could not execute R to verify these directly):
#'   - `coef$copulas` is a dp x 2 matrix of NATURAL copula parameters
#'     (cpar1, cpar2), i.e. already passed through linkCop(), and cpar1/
#'     cpar2 are constant per margin (no copula-level covariates), matching
#'     how EstBifactorTwoStep() itself fits them.
#'   - `coef$margins` is a dp x 2 matrix in NATURAL units (via
#'     link2parM()), NA-padded in column 2 for 1-parameter margins.
#'   - `model` has the same shape used throughout (Sim2Cluster/
#'     EstBifactorTwoStep): model$Y[[j]]$margin$family/dfM,
#'     model$Y[[j]]$copula[[1|2]]$family/rot/dfC, and likewise model$X[[j]].

#' @title (internal) Load or compile the cond_median_logh() C kernel
#' @description Same pattern as EstBifactorCopulaGAMMFast()'s
#' `.load_bifactor_kernel()`: loads `bifactor_kernel.so`/`.dll` if already
#' built and up to date, otherwise compiles `bifactor_kernel.c` with
#' `R CMD SHLIB` (trying `-fopenmp` first, falling back to a
#' single-threaded build). This is the SAME source file/compiled object
#' `EstBifactorCopulaGAMMFast()` already uses for `bifactor_negloglik()`
#' -- `cond_median_logh()` was added alongside it, reusing the same
#' `pcond_c()`/`dcop_c()` static helpers, so loading one gives you both.
#' If you've already called `.load_bifactor_kernel()` (e.g. via
#' `EstBifactorCopulaGAMMFast()`/`EstBifactorTwoStep()`) in this session,
#' this is a near-instant no-op (`is.loaded()` short-circuits it).
#' @keywords internal
.load_cond_median_kernel = function(kernel_path = "bifactor_kernel.c") {
  if (is.loaded("cond_median_logh")) return(invisible(TRUE))
  if (!file.exists(kernel_path)) return(invisible(FALSE))

  if (grepl("\\.c$", kernel_path)) {
    dir  = normalizePath(dirname(kernel_path), mustWork = TRUE)
    file = basename(kernel_path)
    so_name = paste0(tools::file_path_sans_ext(file), .Platform$dynlib.ext)
    so_path = file.path(dir, so_name)

    need_compile = !file.exists(so_path) ||
      file.mtime(file.path(dir, file)) > file.mtime(so_path)

    if (need_compile) {
      wd = getwd(); setwd(dir); on.exit(setwd(wd), add = TRUE)

      build_with = function(cflags, libs) {
        old_cflags = Sys.getenv("PKG_CFLAGS", unset = NA)
        old_libs   = Sys.getenv("PKG_LIBS", unset = NA)
        on.exit({
          if (is.na(old_cflags)) Sys.unsetenv("PKG_CFLAGS") else Sys.setenv(PKG_CFLAGS = old_cflags)
          if (is.na(old_libs))   Sys.unsetenv("PKG_LIBS")   else Sys.setenv(PKG_LIBS = old_libs)
        }, add = TRUE)
        Sys.setenv(PKG_CFLAGS = cflags, PKG_LIBS = libs)
        if (file.exists(so_name)) file.remove(so_name)
        system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", shQuote(file)),
                stdout = TRUE, stderr = TRUE)
      }

      res = build_with("-fopenmp -O2", "-fopenmp")
      if (!file.exists(so_name)) res = build_with("-O2", "")
      if (!file.exists(so_name)) return(invisible(FALSE))
    }
    dyn.load(so_path)
  } else {
    dyn.load(kernel_path)
  }
  invisible(is.loaded("cond_median_logh"))
}

#' @title (internal) Copula family name -> integer code, for the C kernel
#' @description Local copy of the same mapping `family_code()` uses in
#' `EstMargin1Factor.R`/`EstBifactorCopulaGAMMFast.R` -- must match
#' `bifactor_kernel.c`'s family codes exactly.
#' @keywords internal
.family_code = function(fam) {
  fam = ifelse(fam == "normal", "gaussian", fam)
  codes = c(gaussian = 0L, t = 1L, clayton = 2L, frank = 3L, gumbel = 4L,
            joe = 5L, plackett = 6L, galambos = 7L, `huesler-reiss` = 8L,
            fgm = 9L, ind = 10L)
  unname(codes[fam])
}

#' @title (internal) CDF (and, for discrete families, F(y-)) from natural-unit margin parameters
#' @description Mirrors qmargin.R's family dispatch/parameterization, but
#' returns the CDF instead of the quantile -- needed because the fitted
#' margin parameters are already in natural units (post-link2parM()),
#' whereas margins0() expects the raw pre-link theta.
#' @keywords internal
.pmargin_natural = function(y, family, param, dfM = 0) {
  F2 = NA
  switch(family,
    "bernoulli" = , "binomial" = {
      p = param[1]
      F1 = ifelse(y >= 1, 1, 1 - p)
      F2 = ifelse(y >= 1, 1 - p, 0)
    },
    "poisson" = {
      mu = param[1]
      F1 = ppois(y, mu)
      F2 = ifelse(y <= 0, 0, ppois(y - 1, mu))
    },
    "nbinom" = {
      mu = param[1]; size = param[2]
      F1 = pnbinom(y, size = size, mu = mu)
      F2 = ifelse(y <= 0, 0, pnbinom(y - 1, size = size, mu = mu))
    },
    "nbinom1" = {
      p = param[1]; size = param[2]
      F1 = pnbinom(y, size = size, prob = p)
      F2 = ifelse(y <= 0, 0, pnbinom(y - 1, size = size, prob = p))
    },
    "geometric" = {
      p = param[1]
      F1 = pgeom(y, p)
      F2 = ifelse(y <= 0, 0, pgeom(y - 1, p))
    },
    "exponential" = {
      mean = param[1]
      F1 = 1 - exp(-y / mean)
    },
    "weibull" = {
      F1 = pweibull(y, shape = param[1], scale = param[2])
    },
    "normal" = , "gaussian" = {
      F1 = pnorm(y, mean = param[1], sd = param[2])
    },
    "t" = {
      F1 = pt((y - param[1]) / param[2], df = dfM)
    },
    "laplace" = {
      F1 = plap((y - param[1]) / param[2])
    },
    stop(".pmargin_natural(): unsupported margin family '", family, "'")
  )
  list(F1 = F1, F2 = F2)
}

#' @title Conditional (posterior) median of the cluster-level factor V, for ONE cluster, all margins jointly
#' @description Implements (2.9)-(2.11) for a single cluster: for each
#' candidate v, computes every member's own per-observation density
#' h(y_i,x_i,v) by integrating out that member's own nuisance w (the inner
#' loop, over ALL dp margins jointly), multiplies across the n_k members
#' (the outer product in 2.10), then finds the v at which the resulting
#' posterior CDF equals 0.5 -- using the same quadrature-rescaling +
#' `invfunc()`-bisection trick as `MAP.continuous()`/`MAP.discrete()`.
#' @title (internal) log h_k(v) at a caller-supplied v-grid, for one cluster
#' @description Shared by `condMedianV_bifactor()` and `condMeanV_bifactor()`:
#' tries the compiled `cond_median_logh()` C kernel first, falling back to
#' the vectorized pure-R computation (tiling every (member, v-node) pair,
#' summing logs across margins, log-sum-exp across the w-quadrature nodes)
#' if the kernel isn't available or a family isn't covered by it.
#' @keywords internal
.lden_grid_bifactor = function(vgrid, F1, F2, n_k, dp, cont, fam1, rot1, cpar1, dfC1,
                                fam2, rot2, cpar2, dfC2, nl, wl, use_c, kernel_path) {
  nw = length(nl)

  if (use_c) {
    fam1_code = .family_code(fam1); fam2_code = .family_code(fam2)
    if (!anyNA(fam1_code) && !anyNA(fam2_code)) {
      F2c = if (is.null(F2)) matrix(0, n_k, dp) else F2
      F2c[is.na(F2c)] = 0  # unused for continuous columns, but must be finite for .C()
      return(.C("cond_median_logh",
                F1 = as.double(F1), F2 = as.double(F2c),
                n_k = as.integer(n_k), dp = as.integer(dp),
                cont = as.integer(cont),
                fam1 = as.integer(fam1_code), fam2 = as.integer(fam2_code),
                rot1 = as.integer(rot1), rot2 = as.integer(rot2),
                cpar1 = as.double(cpar1), cpar2 = as.double(cpar2),
                dfC1 = as.double(dfC1), dfC2 = as.double(dfC2),
                vgrid = as.double(vgrid), nvg = as.integer(length(vgrid)),
                nl = as.double(nl), wl = as.double(wl), nqw = as.integer(nw),
                out = double(length(vgrid)))$out)
    }
  }

  # Pure-R fallback (vectorized over (member, v-node) pairs -- see
  # condMedianV_bifactor()'s history for why this beats an R-level loop
  # over v-nodes).
  nvg = length(vgrid)
  N2  = n_k * nvg
  v_tile = rep(vgrid, each = n_k)
  A = matrix(0, N2, dp)
  B = if (any(!cont)) matrix(0, N2, dp) else NULL
  for (j in seq_len(dp)) {
    u1_tile = rep(F1[, j], nvg)
    A[, j] = pcond(u1_tile, v_tile, fam1[j], rot1[j], cpar1[j], dfC1[j])
    if (!cont[j]) {
      u2_tile = rep(F2[, j], nvg)
      B[, j] = pcond(u2_tile, v_tile, fam1[j], rot1[j], cpar1[j], dfC1[j])
    }
  }
  logLL = matrix(0, N2, nw)
  for (j in seq_len(dp)) {
    a1 = rep(A[, j], nw)
    wv = rep(nl, each = N2)
    if (cont[j]) {
      tw = pmax(dcop(a1, wv, fam2[j], rot2[j], cpar2[j], dfC2[j]), 1e-300)
    } else {
      b1 = rep(B[, j], nw)
      t1 = pcond(a1, wv, fam2[j], rot2[j], cpar2[j], dfC2[j])
      t2 = pcond(b1, wv, fam2[j], rot2[j], cpar2[j], dfC2[j])
      tw = pmax(t1 - t2, 1e-100)
    }
    logLL = logLL + matrix(log(tw), nrow = N2)
  }
  lmax_w = apply(logLL, 1, max)
  inner  = as.vector(exp(logLL - lmax_w) %*% wl)
  logH_flat = lmax_w + log(pmax(inner, 1e-300))
  colSums(matrix(logH_flat, nrow = n_k))
}

#' @title Conditional (posterior) median of the cluster-level factor V, for ONE cluster, all margins jointly
#' @description Implements (2.9)-(2.11) for a single cluster: for each
#' candidate v, computes every member's own per-observation density
#' h(y_i,x_i,v) by integrating out that member's own nuisance w (the inner
#' loop, over ALL dp margins jointly), multiplies across the n_k members
#' (the outer product in 2.10), then finds the v at which the resulting
#' posterior CDF equals 0.5 -- using the same quadrature-rescaling +
#' `invfunc()`-bisection trick as `MAP.continuous()`/`MAP.discrete()`.
#' @param F1     n_k x dp matrix: F_j(x_ij) (or G_j(y_ij) for Y-margins) for this cluster's members (columns = margins, Y columns first then X, matching `model`'s order)
#' @param F2     n_k x dp matrix: the left-limit CDF, needed only for discrete margins (columns for continuous margins can be NA)
#' @param cont   logical, length dp: TRUE = continuous (uses copula density), FALSE = discrete (uses copula CDF difference)
#' @param fam1,rot1,cpar1,dfC1  length-dp vectors: copula-1 (links each margin to V) family/rotation/parameter/df
#' @param fam2,rot2,cpar2,dfC2  length-dp vectors: copula-2 (links each margin's V-conditional value to the per-observation nuisance w) family/rotation/parameter/df
#' @param nq     number of Gauss-Legendre-on-(0,1) quadrature nodes, used for BOTH the v-integral and each member's own w-integral (default 31, matching MAP.continuous()/MAP.discrete())
#' @param gl     optional precomputed `statmod::gauss.quad.prob(nq)` result, to avoid recomputing it once per cluster (`condMediansBifactor()` supplies this)
#' @param use_c  logical (default TRUE). If TRUE, tries to use the compiled `cond_median_logh()` C kernel (see `kernel_path`) for the expensive nested member/margin/w-node computation, falling back to the pure-R version automatically if the kernel can't be found/compiled/loaded. Set to FALSE to force the pure-R path (e.g. for debugging).
#' @param kernel_path path to `bifactor_kernel.c` (or its already-compiled `.so`/`.dll`) -- the SAME file `EstBifactorCopulaGAMMFast()` uses, since `cond_median_logh()` was added alongside its existing `bifactor_negloglik()`. Defaults to `"bifactor_kernel.c"` in the working directory.
#' @return scalar: the conditional posterior median of V_k for this cluster
#' @export
condMedianV_bifactor = function(F1, F2 = NULL, cont, fam1, rot1, cpar1, dfC1 = NULL,
                                 fam2, rot2, cpar2, dfC2 = NULL, nq = 31, gl = NULL,
                                 use_c = TRUE, kernel_path = "bifactor_kernel.c") {
  F1 = as.matrix(F1)
  dp  = ncol(F1)
  n_k = nrow(F1)
  if (is.null(dfC1)) dfC1 = rep(0, dp)
  if (is.null(dfC2)) dfC2 = rep(0, dp)
  if (any(!cont) && is.null(F2)) stop("F2 is required when any margin is discrete (cont==FALSE)")
  if (!is.null(F2)) F2 = as.matrix(F2)

  # NOTE (perf): gauss.quad.prob(nq) only depends on nq, not on this
  # cluster's data -- recomputing it separately for every cluster (as the
  # previous version did) is wasted work. Accept it precomputed via `gl`
  # (condMediansBifactor() now does this once, outside the cluster loop);
  # only fall back to computing it here if called standalone.
  if (is.null(gl)) gl = statmod::gauss.quad.prob(nq)
  nl = gl$nodes; wl = gl$weights

  use_c = use_c && isTRUE(tryCatch(.load_cond_median_kernel(kernel_path), error = function(e) FALSE))

  lden_grid = function(vgrid) {
    .lden_grid_bifactor(vgrid, F1, F2, n_k, dp, cont, fam1, rot1, cpar1, dfC1,
                         fam2, rot2, cpar2, dfC2, nl, wl, use_c, kernel_path)
  }

  lden  = lden_grid(nl)
  lmax  = max(lden)
  wprd  = wl * exp(lden - lmax)
  normC = sum(wprd)

  # NOTE on a DIFFERENT risk from the overflow/underflow one above: as dp
  # (or n_k) grows, each extra margin/member is another independent piece
  # of evidence about V, so its posterior gets sharper -- log-sum-exp
  # keeps the arithmetic itself safe, but a fixed, modest nq can still be
  # too coarse a grid to resolve a sufficiently narrow peak (a silent
  # accuracy problem, not a crash). A cheap diagnostic: the effective
  # number of quadrature nodes actually contributing mass, via the
  # normalized weights' effective sample size (ESS = (sum w)^2/sum(w^2)).
  # If only a handful of the nq nodes carry essentially all the mass,
  # the median found via `invfunc()` below may be inaccurate -- increase
  # `nq` (and re-check that the estimate stabilizes) if this warns.
  ess = normC^2 / sum(wprd^2)
  if (ess < min(5, nq / 4)) {
    warning("condMedianV_bifactor(): posterior of V looks sharply peaked ",
            "relative to the quadrature grid (effective nodes ~", round(ess, 1),
            " out of nq=", nq, "). Consider increasing nq -- this can happen ",
            "with a large number of margins (dp) and/or cluster members (n_k).")
  }

  # Posterior CDF of V at x, via the same rescaled-quadrature trick
  # MAP.continuous()/MAP.discrete() use: integral_0^x g(v)dv = x * integral_0^1 g(x*t)dt.
  cdff = function(x, thx) {
    ldenx = lden_grid(x * nl)
    x * sum(wl * exp(ldenx - lmax)) / normC
  }

  invfuncNew(0.5, cdff, 0, lb = 1e-10, ub = 1 - 1e-10, tol = 1e-8)
}

#' @title (internal) Shared bookkeeping for condMediansBifactor()/condMeansBifactor()
#' @description Reconstructs, from `Y`/`X`/`model`/`coef`, everything both
#' wrappers need: per-margin family/rotation/df arrays (Y then X, matching
#' `EstBifactorTwoStep()`'s own convention), the natural-unit copula
#' parameters, and each margin's CDF (`F1`/`F2`) evaluated at every
#' observation. Factored out so the median and mean wrappers don't
#' duplicate this setup.
#' @keywords internal
.prepare_bifactor_data = function(Y, X, clu, model, coef) {
  dismod = c("binomial", "bernoulli", "poisson", "nbinom", "geometric", "multinomial")

  if (is.null(dim(Y))) Y = matrix(Y, ncol = 1)
  d = length(model$Y)
  if (is.null(X)) {
    dp = d
  } else {
    if (is.null(dim(X))) X = matrix(X, ncol = 1)
    p = length(model$X)
    dp = p + d
  }
  XX = as.matrix(cbind(Y, X))

  cont   = rep(TRUE, dp)
  rot1   = rot2 = numeric(dp)
  dfC1   = dfC2 = numeric(dp)
  dfM    = numeric(dp)
  margin = fam1 = fam2 = character(dp)

  .fill = function(j, mod) {
    margin[j] <<- mod$margin$family
    if (!is.null(mod$margin$dfM)) dfM[j] <<- mod$margin$dfM
    if (margin[j] %in% dismod) cont[j] <<- FALSE
    fam1[j] <<- mod$copula[[1]]$family
    fam2[j] <<- mod$copula[[2]]$family
    if (!is.null(mod$copula[[1]]$rot)) rot1[j] <<- mod$copula[[1]]$rot
    if (!is.null(mod$copula[[2]]$rot)) rot2[j] <<- mod$copula[[2]]$rot
    if (!is.null(mod$copula[[1]]$dfC)) dfC1[j] <<- mod$copula[[1]]$dfC
    if (!is.null(mod$copula[[2]]$dfC)) dfC2[j] <<- mod$copula[[2]]$dfC
  }
  for (j in seq_len(d)) .fill(j, model$Y[[j]])
  if (dp > d) for (j in seq_len(p)) .fill(j + d, model$X[[j]])

  cpar1 = coef$copulas[, 1]
  cpar2 = coef$copulas[, 2]

  F1 = matrix(NA_real_, nrow(XX), dp)
  F2 = matrix(NA_real_, nrow(XX), dp)
  for (j in seq_len(dp)) {
    pm = .pmargin_natural(XX[, j], margin[j], coef$margins[j, ], dfM[j])
    F1[, j] = pm$F1
    if (!cont[j]) F2[, j] = pm$F2
  }

  if (is.factor(clu)) clusters = levels(clu) else clusters = unique(clu)

  list(F1 = F1, F2 = F2, cont = cont, fam1 = fam1, fam2 = fam2,
       rot1 = rot1, rot2 = rot2, dfC1 = dfC1, dfC2 = dfC2,
       cpar1 = cpar1, cpar2 = cpar2, clu = clu, clusters = clusters,
       margin = margin, dfM = dfM)
}

#' @title (internal) Batched, all-clusters-at-once setup for condMediansBifactor()/condMeansBifactor()
#' @description Tries to load the `cond_median_logh_multi()` C kernel
#' (batched across ALL K clusters, with per-cluster v-grids, OpenMP-
#' parallelized across clusters -- the multi-cluster companion to
#' `cond_median_logh()`) and, if available, runs the shared "evaluate at
#' the base grid `nl` for every cluster in ONE call" step both the mean
#' and the median need. This is what lets `condMediansBifactor()` call
#' `invfunc()` a single time across all K clusters at once (using
#' `invfunc()`'s own built-in vectorization -- it subsets `q`/`v0`/`v1`/
#' `th` together by the same convergence mask, so passing cluster indices
#' as `th` lets our `func` recover which clusters are still active at each
#' bisection step) instead of K separate per-cluster `invfunc()` calls.
#' Falls back to `list(use_multi = FALSE, ...)` if the multi-cluster
#' kernel isn't available (older compiled kernel, no C compiler, an
#' uncovered family, etc.) -- callers should fall back to the original
#' per-cluster loop over `condMedianV_bifactor()`/`condMeanV_bifactor()`
#' in that case.
#' @keywords internal
.prepare_bifactor_batch = function(prep, nl, wl, nq, use_c, kernel_path) {
  K = length(prep$clusters)
  if (use_c) use_c = isTRUE(tryCatch(.load_cond_median_kernel(kernel_path), error = function(e) FALSE))
  fam1_code = fam2_code = NULL
  use_multi = FALSE
  if (use_c) {
    fam1_code = .family_code(prep$fam1); fam2_code = .family_code(prep$fam2)
    use_c = !anyNA(fam1_code) && !anyNA(fam2_code)
    use_multi = use_c && is.loaded("cond_median_logh_multi")
  }
  if (!use_multi) return(list(use_multi = FALSE, use_c = use_c))

  dp = ncol(prep$F1); N = nrow(prep$F1)
  ord_list = lapply(seq_len(K), function(k) which(prep$clu == prep$clusters[k]))
  order0 = as.integer(unlist(ord_list) - 1L)
  cbound = as.integer(c(0, cumsum(vapply(ord_list, length, integer(1)))))
  F2c = prep$F2; F2c[is.na(F2c)] = 0

  # Batched call: evaluates log h_k(v) for whichever clusters are listed
  # in `active_k0` (0-based), each at its OWN row of `vgrid_mat` -- ONE
  # .C() call regardless of how many clusters are still active.
  call_multi = function(active_k0, vgrid_mat) {
    n_active = length(active_k0)
    .C("cond_median_logh_multi",
       F1 = as.double(prep$F1), F2 = as.double(F2c),
       N = as.integer(N), dp = as.integer(dp),
       cont = as.integer(prep$cont),
       fam1 = as.integer(fam1_code), fam2 = as.integer(fam2_code),
       rot1 = as.integer(prep$rot1), rot2 = as.integer(prep$rot2),
       cpar1 = as.double(prep$cpar1), cpar2 = as.double(prep$cpar2),
       dfC1 = as.double(prep$dfC1), dfC2 = as.double(prep$dfC2),
       order0 = order0, cbound = cbound,
       active_k = as.integer(active_k0), n_active = as.integer(n_active),
       vgrid = as.double(vgrid_mat), nq = as.integer(ncol(vgrid_mat)),
       nl = as.double(nl), wl = as.double(wl), nqw = as.integer(length(nl)),
       out = double(n_active * ncol(vgrid_mat)))$out
  }

  # Base-grid evaluation for ALL K clusters in ONE call (every cluster
  # uses the SAME grid `nl` here -- only the later, per-bisection-step
  # rescaled grids differ by cluster).
  vgrid0 = matrix(rep(nl, each = K), nrow = K)
  lden0  = matrix(call_multi(0:(K - 1), vgrid0), nrow = K)
  lmax_k = apply(lden0, 1, max)
  wprd_k = sweep(exp(lden0 - lmax_k), 2, wl, `*`)
  normC_k = rowSums(wprd_k)

  # Same ESS diagnostic as condMedianV_bifactor()/condMeanV_bifactor(),
  # vectorized across all K clusters at once.
  ess_k = normC_k^2 / rowSums(wprd_k^2)
  low = which(ess_k < min(5, nq / 4))

  list(use_multi = TRUE, use_c = use_c, call_multi = call_multi,
       lmax_k = lmax_k, normC_k = normC_k, wprd_k = wprd_k, low = low)
}

#' @title Conditional median of V for every cluster, all Y/X components jointly
#' @description Top-level wrapper: given the original data and a fitted
#' bi-factor model (as returned by `EstBifactorTwoStep()`/
#' `EstBifactorCopulaGAMMFast()`), reconstructs each margin's CDF from its
#' fitted (natural-unit) parameters, then estimates every cluster's V_k.
#'
#' NOTE (perf): when the batched `cond_median_logh_multi()` C kernel is
#' available, this calls `invfunc()` ONCE across ALL K clusters
#' simultaneously (see `.prepare_bifactor_batch()`), so each of its ~40
#' bisection steps costs a single `.C()` call spanning whichever clusters
#' haven't converged yet -- instead of the ~40*K small `.C()` calls the
#' original per-cluster loop made. Falls back automatically to that
#' original per-cluster loop (via `condMedianV_bifactor()`, still using
#' the single-cluster kernel if available) when the multi-cluster kernel
#' can't be used.
#' @param Y,X    N x d / N x p matrices (or vectors) of response/covariate observations; X may be NULL
#' @param clu    cluster membership, length N
#' @param model  same model spec as `Sim2Cluster()`/`EstBifactorTwoStep()`
#' @param coef   the fitted `coef` list (`$margins`, `$copulas`) -- see the ASSUMPTIONS note at the top of this file re: natural units
#' @param nq     number of quadrature nodes (default 31)
#' @param use_c  logical (default TRUE), tries the compiled kernel(s) (batched multi-cluster first, then single-cluster), falling back to pure R
#' @param kernel_path path to `bifactor_kernel.c`/`.so`
#' @return data.frame with one row per cluster: `cluster`, `V` (conditional posterior median)
#' @export
condMediansBifactor = function(Y, X = NULL, clu, model, coef, nq = 31,
                                use_c = TRUE, kernel_path = "bifactor_kernel.c") {
  prep = .prepare_bifactor_data(Y, X, clu, model, coef)
  K = length(prep$clusters)
  gl = statmod::gauss.quad.prob(nq)
  nl = gl$nodes; wl = gl$weights

  batch = .prepare_bifactor_batch(prep, nl, wl, nq, use_c, kernel_path)

  if (batch$use_multi) {
    if (length(batch$low) > 0) {
      warning("condMediansBifactor(): posterior of V looks sharply peaked for ",
              length(batch$low), " cluster(s) (e.g. cluster ", prep$clusters[batch$low[1]],
              "). Consider increasing nq.")
    }

    # `func` for invfunc(): x_vec is the current bisection point for each
    # still-active cluster, and k_idx is that SAME subset's cluster
    # indices (recovered from invfunc()'s own consistent subsetting of
    # `th`, which we set to 1:K below) -- so this one function call
    # batches the C kernel across every cluster invfunc() still has open.
    batched_cdff = function(x_vec, k_idx) {
      k0 = as.integer(round(k_idx)) - 1L
      vgrid = outer(x_vec, nl)
      outm  = matrix(batch$call_multi(k0, vgrid), nrow = length(x_vec))
      lmax_sel  = batch$lmax_k[k0 + 1L]
      normC_sel = batch$normC_k[k0 + 1L]
      x_vec * rowSums(sweep(exp(outm - lmax_sel), 2, wl, `*`)) / normC_sel
    }

    Vhat = invfuncNew(rep(0.5, K), batched_cdff, 1:K, lb = 1e-10, ub = 1 - 1e-10, tol = 1e-8)
    return(data.frame(cluster = prep$clusters, V = Vhat))
  }

  # Fallback: per-cluster loop (still uses the single-cluster C kernel,
  # cond_median_logh(), when available -- see condMedianV_bifactor()).
  Vhat = rep(NA_real_, K)
  for (k in seq_len(K)) {
    idx = which(prep$clu == prep$clusters[k])
    Vhat[k] = condMedianV_bifactor(prep$F1[idx, , drop = FALSE], prep$F2[idx, , drop = FALSE],
                                    prep$cont, prep$fam1, prep$rot1, prep$cpar1, prep$dfC1,
                                    prep$fam2, prep$rot2, prep$cpar2, prep$dfC2, nq, gl = gl,
                                    use_c = batch$use_c, kernel_path = kernel_path)
  }

  data.frame(cluster = prep$clusters, V = Vhat)
}

#' @title Conditional (posterior) MEAN of the cluster-level factor V, for ONE cluster, all margins jointly
#' @description Companion to `condMedianV_bifactor()`, computing the
#' conditional posterior MEAN of V_k instead of its median:
#' `E[V_k | data] = integral v * h_k(v) dv / integral h_k(v) dv`. This is
#' substantially cheaper than the median, since it only needs `h_k(v)`
#' evaluated ONCE, at the base quadrature grid -- unlike the median, which
#' needs `invfunc()`'s ~40 bisection evaluations, each at a different
#' rescaled grid. Reuses the exact same `.lden_grid_bifactor()` machinery
#' (and, when available, the compiled `cond_median_logh()` C kernel) as
#' `condMedianV_bifactor()` -- source that part of the file first.
#' @inheritParams condMedianV_bifactor
#' @return scalar: the conditional posterior mean of V_k for this cluster
#' @export
condMeanV_bifactor = function(F1, F2 = NULL, cont, fam1, rot1, cpar1, dfC1 = NULL,
                               fam2, rot2, cpar2, dfC2 = NULL, nq = 31, gl = NULL,
                               use_c = TRUE, kernel_path = "bifactor_kernel.c") {
  F1 = as.matrix(F1)
  dp  = ncol(F1)
  n_k = nrow(F1)
  if (is.null(dfC1)) dfC1 = rep(0, dp)
  if (is.null(dfC2)) dfC2 = rep(0, dp)
  if (any(!cont) && is.null(F2)) stop("F2 is required when any margin is discrete (cont==FALSE)")
  if (!is.null(F2)) F2 = as.matrix(F2)

  if (is.null(gl)) gl = statmod::gauss.quad.prob(nq)
  nl = gl$nodes; wl = gl$weights

  use_c = use_c && isTRUE(tryCatch(.load_cond_median_kernel(kernel_path), error = function(e) FALSE))

  # Only ONE evaluation needed -- no bisection at all.
  lden = .lden_grid_bifactor(nl, F1, F2, n_k, dp, cont, fam1, rot1, cpar1, dfC1,
                              fam2, rot2, cpar2, dfC2, nl, wl, use_c, kernel_path)
  lmax  = max(lden)
  wprd  = wl * exp(lden - lmax)
  normC = sum(wprd)

  # Same ESS diagnostic as condMedianV_bifactor(): the mean is a
  # quadrature-weighted average, so it's just as vulnerable to a
  # too-coarse grid missing a sharply peaked posterior.
  ess = normC^2 / sum(wprd^2)
  if (ess < min(5, nq / 4)) {
    warning("condMeanV_bifactor(): posterior of V looks sharply peaked ",
            "relative to the quadrature grid (effective nodes ~", round(ess, 1),
            " out of nq=", nq, "). Consider increasing nq.")
  }

  sum(nl * wprd) / normC
}

#' @title Conditional mean of V for every cluster, all Y/X components jointly
#' @description Same interface/bookkeeping as `condMediansBifactor()`. When
#' the batched `cond_median_logh_multi()` kernel is available, computes
#' every cluster's mean from the SAME single, all-K-clusters-at-once
#' evaluation `.prepare_bifactor_batch()` performs (the mean needs no
#' bisection, so this is just one weighted average per cluster after that
#' one shared call) -- otherwise falls back to the original per-cluster
#' loop over `condMeanV_bifactor()`.
#' @inheritParams condMediansBifactor
#' @return data.frame with one row per cluster: `cluster`, `V` (conditional posterior mean)
#' @export
condMeansBifactor = function(Y, X = NULL, clu, model, coef, nq = 31,
                              use_c = TRUE, kernel_path = "bifactor_kernel.c") {
  prep = .prepare_bifactor_data(Y, X, clu, model, coef)
  K = length(prep$clusters)
  gl = statmod::gauss.quad.prob(nq)
  nl = gl$nodes; wl = gl$weights

  batch = .prepare_bifactor_batch(prep, nl, wl, nq, use_c, kernel_path)

  if (batch$use_multi) {
    if (length(batch$low) > 0) {
      warning("condMeansBifactor(): posterior of V looks sharply peaked for ",
              length(batch$low), " cluster(s) (e.g. cluster ", prep$clusters[batch$low[1]],
              "). Consider increasing nq.")
    }
    Vhat = as.vector(batch$wprd_k %*% nl) / batch$normC_k
    return(data.frame(cluster = prep$clusters, V = Vhat))
  }

  # Fallback: per-cluster loop (still uses the single-cluster C kernel,
  # cond_median_logh(), when available -- see condMeanV_bifactor()).
  Vhat = rep(NA_real_, K)
  for (k in seq_len(K)) {
    idx = which(prep$clu == prep$clusters[k])
    Vhat[k] = condMeanV_bifactor(prep$F1[idx, , drop = FALSE], prep$F2[idx, , drop = FALSE],
                                  prep$cont, prep$fam1, prep$rot1, prep$cpar1, prep$dfC1,
                                  prep$fam2, prep$rot2, prep$cpar2, prep$dfC2, nq, gl = gl,
                                  use_c = batch$use_c, kernel_path = kernel_path)
  }

  data.frame(cluster = prep$clusters, V = Vhat)
}
