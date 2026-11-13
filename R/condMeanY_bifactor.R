#' condMeanY_bifactor.R
#'
#' Computes the conditional MEAN of each Y-component Y_j (j=1..d), given
#' the observed X covariates of each observation, following "Bi-factor
#' copulas for clustered data":
#'
#'   - condMeanY_given_XV(): E[Y_j | X=x, V=v], plugging in a PREDICTED
#'     value of the cluster-level latent factor V (e.g. from
#'     condMeansBifactor()/condMediansBifactor()) -- continuous Y_j uses
#'     eq. (2.5), discrete Y_j uses eq. (2.6).
#'   - condMeanY_given_X(): E[Y_j | X=x], integrating V out entirely (no
#'     latent-variable estimate needed at all) -- continuous Y_j uses eq.
#'     (2.10), discrete Y_j uses eq. (2.11).
#'
#' Both follow the `newdata = NULL` convention already used by
#' `predictContinuous.R`: `newdata = NULL` re-predicts the original
#' training observations; otherwise `newdata` is a list with an `x`
#' element (and, for `condMeanY_given_XV()`, a `clu` element saying which
#' fitted cluster's V to use for each new row).
#'
#' Source `condMedianV_bifactor.R` first -- this file reuses
#' `.prepare_bifactor_data()`, `.pmargin_natural()`, and the package's own
#' `pcond()`/`dcop()`/`qmargin()` (make sure you're using the FIXED
#' `qmargin.R`, which this file also depends on for continuous Y margins
#' other than normal/t/laplace -- the original had a bug in five of its
#' branches; see that file's own NOTE).
#'
#' MATH RECAP (see the paper for the full derivation): both eq. (2.5)/
#' (2.6) and (2.10)/(2.11) share the same "X-side" building block --
#' B(x,v) = integral_0^1 [product over X-margins l of the copula-1 x
#' copula-2 contribution] dw -- which is EXACTLY the same quantity our
#' `cond_median_logh()`/`.lden_grid_bifactor()` machinery already computes
#' as "h_obs" for one observation, just restricted to the X-margins only
#' (no Y-term). `.Xw_terms()`/`.EYj_terms_given_Xv()` below recompute this
#' directly in R (rather than reusing that C-kernel path), since here we
#' additionally need the RAW per-w-node values (not already collapsed by
#' quadrature) to combine with each Y-quantile/Y-mass term -- this is a
#' one-off per-row prediction, not a hot loop inside estimation, so plain
#' (not C-accelerated) R is used throughout.

#' @title (internal) X-side w-integrand, one observation, at a fixed v
#' @description Computes, for ONE observation's X covariates (already
#' transformed to F_l(x_l)/F_l(x_l-)) and a fixed v, the vector (over
#' `wgrid`) of `product over l of [c_l,1[F_l(x_l),v] * c_l,2[C_l,1[F_l(x_l),v],w]]`
#' (continuous X_l) or `[C_l,2[...,w] - C_l,2[...,w]]` (discrete X_l) --
#' i.e. the integrand of B(x,v) (eqs. 2.3/2.8) before the w-quadrature sum.
#' @keywords internal
.Xw_terms = function(x_F1, x_F2, cont_X, fam1_X, rot1_X, cpar1_X, dfC1_X,
                      fam2_X, rot2_X, cpar2_X, dfC2_X, v, wgrid) {
  p  = length(fam1_X)
  nw = length(wgrid)
  out = rep(1, nw)
  if (p == 0) return(out)  # no X covariates: B(x,v) degenerates to a constant
  for (l in seq_len(p)) {
    A = pcond(x_F1[l], v, fam1_X[l], rot1_X[l], cpar1_X[l], dfC1_X[l])
    if (cont_X[l]) {
      cl1  = dcop(x_F1[l], v, fam1_X[l], rot1_X[l], cpar1_X[l], dfC1_X[l])
      term = cl1 * dcop(rep(A, nw), wgrid, fam2_X[l], rot2_X[l], cpar2_X[l], dfC2_X[l])
    } else {
      B  = pcond(x_F2[l], v, fam1_X[l], rot1_X[l], cpar1_X[l], dfC1_X[l])
      t1 = pcond(rep(A, nw), wgrid, fam2_X[l], rot2_X[l], cpar2_X[l], dfC2_X[l])
      t2 = pcond(rep(B, nw), wgrid, fam2_X[l], rot2_X[l], cpar2_X[l], dfC2_X[l])
      term = pmax(t1 - t2, 1e-100)
    }
    out = out * pmax(term, 1e-300)
  }
  out
}

#' @title (internal) E[Y_j] numerator and B(x,v), for ONE observation, ONE Y-margin, fixed v
#' @description Returns `list(Bxv, Enum)` such that `E[Y_j|X=x,V=v] = Enum/Bxv`
#' (eq. 2.5 for continuous Y_j, eq. 2.6 for discrete Y_j). For the
#' integrated-over-V functions, callers combine several v-nodes'
#' `Bxv`/`Enum` via a further v-quadrature (eqs. 2.10/2.11) before
#' dividing, rather than dividing here and re-weighting afterward.
#' @keywords internal
.EYj_terms_given_Xv = function(v, x_F1, x_F2, cont_X, fam1_X, rot1_X, cpar1_X, dfC1_X,
                                fam2_X, rot2_X, cpar2_X, dfC2_X,
                                margin_j, param_j, dfM_j,
                                fam1_j, rot1_j, cpar1_j, dfC1_j,
                                fam2_j, rot2_j, cpar2_j, dfC2_j,
                                is_disc_j, nl, wl) {
  nw = length(wl)
  Xw  = .Xw_terms(x_F1, x_F2, cont_X, fam1_X, rot1_X, cpar1_X, dfC1_X,
                   fam2_X, rot2_X, cpar2_X, dfC2_X, v, nl)
  Bxv = sum(wl * Xw)

  if (!is_disc_j) {
    # eq. (2.5): continuous Y_j -- integrate Gj^{-1}(u) against the
    # (unnormalized) conditional density of U=Gj(Yj) given X=x,V=v.
    vrep = rep(v, length(nl))
    Aj   = pcond(nl, vrep, fam1_j, rot1_j, cpar1_j, dfC1_j)   # Dj,1(u,v), per u-node
    dj1  = dcop(nl, vrep, fam1_j, rot1_j, cpar1_j, dfC1_j)    # dj,1(u,v), per u-node
    inner = vapply(seq_along(nl), function(iu) {
      sum(wl * dcop(rep(Aj[iu], nw), nl, fam2_j, rot2_j, cpar2_j, dfC2_j) * Xw)
    }, numeric(1))
    dens = dj1 * inner
    yq   = qmargin(nl, margin_j, param_j, dfM = dfM_j)
    Enum = sum(wl * yq * dens)
  } else {
    # eq. (2.6): discrete Y_j -- sum y*P(Yj=y|X=x,V=v) over the (truncated) support.
    ymax = if (margin_j %in% c("bernoulli", "binomial")) {
      1L  # exact support {0,1} -- no need to estimate a truncation bound
    } else {
      ceiling(qmargin(0.99999, margin_j, param_j, dfM = dfM_j)) + 5
    }
    ys = 0:ymax
    pm = .pmargin_natural(ys, margin_j, param_j, dfM_j)   # F1=Gj(y), F2=Gj(y-)
    Enum = 0
    for (iy in seq_along(ys)) {
      Ajy   = pcond(pm$F1[iy], v, fam1_j, rot1_j, cpar1_j, dfC1_j)
      Ajym  = pcond(pm$F2[iy], v, fam1_j, rot1_j, cpar1_j, dfC1_j)
      massw = pcond(rep(Ajy, nw), nl, fam2_j, rot2_j, cpar2_j, dfC2_j) -
              pcond(rep(Ajym, nw), nl, fam2_j, rot2_j, cpar2_j, dfC2_j)
      mass = sum(wl * pmax(massw, 1e-100) * Xw)
      Enum = Enum + ys[iy] * mass
    }
  }
  list(Bxv = Bxv, Enum = Enum)
}

#' @title Conditional mean of every Y-component given X and a predicted V, for each observation
#' @description E[Y_j | X=x, V=v] for j=1..d, for every observation, using
#' each observation's OWN X covariates and its CLUSTER's predicted latent
#' factor V (from `condMeansBifactor()`/`condMediansBifactor()`) -- eq.
#' (2.5) for continuous Y_j, eq. (2.6) for discrete Y_j.
#' @param Y,X    original training data (as passed to `EstBifactorTwoStep()`); only used (and only needed) when `newdata=NULL`
#' @param clu    original training cluster membership; only used when `newdata=NULL`
#' @param model  same model spec as `Sim2Cluster()`/`EstBifactorTwoStep()`
#' @param coef   the fitted `coef` list (`$margins`, `$copulas`)
#' @param Vhat   data.frame with `cluster`/`V` columns, as returned by `condMeansBifactor()`/`condMediansBifactor()` -- looked up by cluster label for each row
#' @param newdata NULL (default) to re-predict the training data, or a list with `x` (new X covariates, matrix/vector) and `clu` (which fitted cluster's V to use for each new row -- must already appear in `Vhat$cluster`)
#' @param nq     number of quadrature nodes, used for BOTH the Y-quantile grid and the w-integral (default 31)
#' @return data.frame, one row per observation, columns `EY1`,...,`EYd`
#' @export
condMeanY_given_XV = function(Y = NULL, X = NULL, clu = NULL, model, coef, Vhat,
                               newdata = NULL, nq = 31) {
  d = length(model$Y)

  if (is.null(newdata)) {
    Xuse = X; cluUse = clu
  } else {
    Xuse = newdata$x; cluUse = newdata$clu
  }
  if (is.null(dim(Xuse))) Xuse = matrix(Xuse, ncol = 1)
  n = nrow(Xuse)

  # Reuses .prepare_bifactor_data()'s bookkeeping/CDF machinery; the Y
  # argument there is only used to build F1/F2 for the Y-margins (which
  # this function doesn't need, since it's PREDICTING Y, not conditioning
  # on it), so a placeholder of the right shape is passed and its
  # Y-derived columns are simply never read below.
  prep = .prepare_bifactor_data(matrix(0, n, d), Xuse, rep(1, n), model, coef)
  p  = ncol(Xuse); dp = d + p
  Xidx = if (dp > d) (d + 1):dp else integer(0)

  gl = statmod::gauss.quad.prob(nq); nl = gl$nodes; wl = gl$weights

  Vlookup = Vhat$V
  names(Vlookup) = as.character(Vhat$cluster)
  vvec = Vlookup[as.character(cluUse)]
  if (anyNA(vvec)) {
    stop("condMeanY_given_XV(): some cluster label(s) in `clu`/`newdata$clu` ",
         "are not present in `Vhat$cluster`.")
  }

  out = matrix(NA_real_, n, d)
  for (i in seq_len(n)) {
    v = as.numeric(vvec[i])
    xF1 = if (p > 0) prep$F1[i, Xidx] else numeric(0)
    xF2 = if (p > 0) prep$F2[i, Xidx] else numeric(0)
    for (j in seq_len(d)) {
      te = .EYj_terms_given_Xv(
        v, xF1, xF2,
        prep$cont[Xidx], prep$fam1[Xidx], prep$rot1[Xidx], prep$cpar1[Xidx], prep$dfC1[Xidx],
        prep$fam2[Xidx], prep$rot2[Xidx], prep$cpar2[Xidx], prep$dfC2[Xidx],
        margin_j = prep$margin[j], param_j = coef$margins[j, ], dfM_j = prep$dfM[j],
        fam1_j = prep$fam1[j], rot1_j = prep$rot1[j], cpar1_j = prep$cpar1[j], dfC1_j = prep$dfC1[j],
        fam2_j = prep$fam2[j], rot2_j = prep$rot2[j], cpar2_j = prep$cpar2[j], dfC2_j = prep$dfC2[j],
        is_disc_j = !prep$cont[j], nl = nl, wl = wl
      )
      out[i, j] = te$Enum / te$Bxv
    }
  }
  out = as.data.frame(out)
  names(out) = paste0("EY", seq_len(d))
  out
}

#' @title Conditional mean of every Y-component given X only (V integrated out), for each observation
#' @description E[Y_j | X=x] for j=1..d, for every observation, using each
#' observation's OWN X covariates and integrating the cluster-level latent
#' factor V out entirely -- no predicted V needed at all (and hence no
#' `clu`/`Vhat` either). Eq. (2.10) for continuous Y_j, eq. (2.11) for
#' discrete Y_j.
#' @param X      original training covariates; only used (and only needed) when `newdata=NULL`
#' @param model  same model spec as `Sim2Cluster()`/`EstBifactorTwoStep()`
#' @param coef   the fitted `coef` list (`$margins`, `$copulas`)
#' @param newdata NULL (default) to re-predict the training X's, or a list with `x` (new X covariates, matrix/vector)
#' @param nq     number of quadrature nodes for v, w, AND the Y-quantile grid (default 25 -- this function needs one more nested integral than `condMeanY_given_XV()`, so a smaller default keeps it from being needlessly slow; raise it if precision looks off)
#' @return data.frame, one row per observation, columns `EY1`,...,`EYd`
#' @export
condMeanY_given_X = function(X = NULL, model, coef, newdata = NULL, nq = 25) {
  d = length(model$Y)

  Xuse = if (is.null(newdata)) X else newdata$x
  if (is.null(dim(Xuse))) Xuse = matrix(Xuse, ncol = 1)
  n = nrow(Xuse)

  prep = .prepare_bifactor_data(matrix(0, n, d), Xuse, rep(1, n), model, coef)
  p  = ncol(Xuse); dp = d + p
  Xidx = if (dp > d) (d + 1):dp else integer(0)

  gl = statmod::gauss.quad.prob(nq); nl = gl$nodes; wl = gl$weights

  out = matrix(NA_real_, n, d)
  for (i in seq_len(n)) {
    xF1 = if (p > 0) prep$F1[i, Xidx] else numeric(0)
    xF2 = if (p > 0) prep$F2[i, Xidx] else numeric(0)
    for (j in seq_len(d)) {
      is_disc_j = !prep$cont[j]
      Bv = numeric(length(nl)); Ev = numeric(length(nl))
      for (iv in seq_along(nl)) {
        te = .EYj_terms_given_Xv(
          nl[iv], xF1, xF2,
          prep$cont[Xidx], prep$fam1[Xidx], prep$rot1[Xidx], prep$cpar1[Xidx], prep$dfC1[Xidx],
          prep$fam2[Xidx], prep$rot2[Xidx], prep$cpar2[Xidx], prep$dfC2[Xidx],
          margin_j = prep$margin[j], param_j = coef$margins[j, ], dfM_j = prep$dfM[j],
          fam1_j = prep$fam1[j], rot1_j = prep$rot1[j], cpar1_j = prep$cpar1[j], dfC1_j = prep$dfC1[j],
          fam2_j = prep$fam2[j], rot2_j = prep$rot2[j], cpar2_j = prep$cpar2[j], dfC2_j = prep$dfC2[j],
          is_disc_j = is_disc_j, nl = nl, wl = wl
        )
        Bv[iv] = te$Bxv; Ev[iv] = te$Enum
      }
      # eqs. (2.10)/(2.11): B(x) and the Y-mean's numerator are each the
      # OUTER v-integral of Bxv(v)/Enum(v) -- combine via the SAME
      # quadrature nodes/weights used for v throughout this file.
      Bx = sum(wl * Bv)
      out[i, j] = sum(wl * Ev) / Bx
    }
  }
  out = as.data.frame(out)
  names(out) = paste0("EY", seq_len(d))
  out
}

#' @title Conditional mean of every Y-component given X, with or without a predicted V
#' @description Single entry point combining `condMeanY_given_XV()` and
#' `condMeanY_given_X()`: if `V` is supplied, computes E[Y_j|X=x,V=v]
#' (eqs. 2.5/2.6), plugging in each row's cluster's predicted latent
#' factor; if `V=NULL` (default), computes E[Y_j|X=x] instead (eqs.
#' 2.10/2.11), integrating the latent factor out entirely -- in that case
#' `clu` isn't needed either (silently ignored if supplied).
#' @param Y,X    original training data; only used (and only needed) when `newdata=NULL`
#' @param clu    original training cluster membership; only used when `newdata=NULL` AND `V` is supplied
#' @param model  same model spec as `Sim2Cluster()`/`EstBifactorTwoStep()`
#' @param coef   the fitted `coef` list (`$margins`, `$copulas`)
#' @param V      NULL (default), or a data.frame with `cluster`/`V` columns (as returned by `condMeansBifactor()`/`condMediansBifactor()`) to condition on -- see `condMeanY_given_XV()`
#' @param newdata NULL (default) to re-predict the training data, or a list with `x` (and, when `V` is supplied, `clu`) -- see the two underlying functions
#' @param nq     number of quadrature nodes. Defaults to 31 when `V` is supplied (`condMeanY_given_XV()`'s default) or 25 when it isn't (`condMeanY_given_X()`'s default, since that path needs one more nested integral); pass a value explicitly to override either.
#' @return data.frame, one row per observation, columns `EY1`,...,`EYd`
#' @export
condMeanY = function(Y = NULL, X = NULL, clu = NULL, model, coef, V = NULL,
                      newdata = NULL, nq = NULL) {
  if (is.null(V)) {
    condMeanY_given_X(X = X, model = model, coef = coef, newdata = newdata,
                       nq = if (is.null(nq)) 25 else nq)
  } else {
    condMeanY_given_XV(Y = Y, X = X, clu = clu, model = model, coef = coef, Vhat = V,
                        newdata = newdata, nq = if (is.null(nq)) 31 else nq)
  }
}
