#' @title Step 1 of the two-step bi-factor estimator: single-margin, 1-factor fit
#'
#' @description Estimates, for one response or covariate margin j in
#' isolation, its marginal-distribution parameters and the parameter of
#' the copula linking it to the cluster-level latent factor V, using only
#' the univariate 1-factor clustered-data likelihood of eq. (2.6)/(2.7) of
#' the bi-factor copula paper -- i.e. every other margin and the
#' second-level latent factor W are ignored entirely. This is exactly
#' Step 1 of the two-step estimator described in that section: "the
#' parameters of Gj and Dj,1 can be estimated using the CRAN package
#' CopulaGAMM ... with no covariates," applied separately to each margin.
#'
#' Compared to plugging this margin into the full bi-factor joint
#' likelihood, this is cheap in two independent ways: (1) it needs only a
#' *single* Gauss-Legendre quadrature over V (no integral over W at all),
#' and (2) it only has `nth+1` parameters (the margin's own 1-2 parameters
#' plus one copula parameter) instead of the full model's parameter count,
#' so even a finite-difference gradient over just this piece is cheap.
#'
#' Where possible, the gradient used by `nlm()` is *analytic*, built from
#' the closed-form derivatives already implemented in `margins()` (margin
#' cdf/pdf and their parameter derivatives) and `coplik()` (copula
#' conditional-cdf/density and their derivatives), combined via the chain
#' rule -- the same style `mlecop()` in `coplik.R` already uses for the
#' simple i.i.d. case, extended here to the clustered/latent-factor
#' likelihood (a quadrature-weighted mixture over V).
#'
#' As with `EstBifactorCopulaGAMMFast()`, the per-cluster/quadrature
#' reduction (previously an R-level loop) is offloaded to a compiled C
#' kernel, `margin1factor_negloglik_grad()` in `margin1factor_kernel.c`,
#' whenever `cop1` is one of the families `coplik()` covers. That kernel
#' also parallelizes across clusters with OpenMP (clusters are
#' independent here too), the same way the joint model's kernel does.
#' `margins()` itself still runs in R (a single vectorized call per
#' evaluation, not a per-cluster loop, so porting every margin family to
#' C would cost more than it saves); only the reduction and the
#' copula-side derivatives are in C. Falls back automatically to the
#' pure-R evaluator below if the kernel can't be found/compiled/loaded.
#'
#' Coverage note: `coplik()` provides analytic derivatives for copula
#' families "gaussian"/"normal", "t", "clayton", "frank", "gumbel", "joe",
#' "plackett", "fgm". It does not cover "galambos", "huesler-reiss", or
#' "ind". For a continuous margin with one of the covered families, the
#' gradient below is fully analytic (both the margin-parameter and the
#' copula-parameter pieces). For a discrete margin, `coplik()` does not
#' expose an analytic d/du of the conditional cdf C(u|v) (needed for the
#' margin-parameter piece via the chain rule through u=G(y;theta)), so
#' that piece falls back to a cheap central difference -- still only 1-2
#' extra copula evaluations, since this is a 2-3 parameter problem, not a
#' finite-difference gradient over the whole model. For a copula family
#' not covered by `coplik()` at all, the whole gradient falls back to
#' `nlm()`'s own finite-difference gradient (via `dcop()`/`pcond()`),
#' which is still cheap here for the same reason.
#'
#' @param z      the N observations for this single margin (one column of
#'   Y or X).
#' @param clu    cluster membership, length N (same convention as
#'   `EstBifactorCopulaGAMM`).
#' @param margin margin family string, as used by `margins()`/`margins0()`.
#' @param cop1   copula-1 family string, as used by `coplik()`/`dcop()`/`pcond()`.
#' @param rot1   rotation for copula 1 (0/90/180/270, default 0).
#' @param dfM    Student margin degrees of freedom (0 if not applicable).
#' @param dfC1   Student copula-1 degrees of freedom (0 if not applicable).
#' @param nq     number of Gauss-Legendre-on-(0,1) quadrature nodes (default 25).
#' @param start  starting values, `c(margin params..., cpar1)`.
#' @param LB,UB  bounds, same length/order as `start`.
#' @param kernel_path Path to the compiled Step-1 C kernel
#'   (`margin1factor_kernel.c`, or its already-compiled `.so`/`.dll`).
#'   Defaults to `"margin1factor_kernel.c"` in the current working
#'   directory; compiled automatically (with an OpenMP attempt, same as
#'   `EstBifactorCopulaGAMMFast()`'s kernel) the first time it's needed.
#'   Set to `NULL` to force the pure-R evaluator (e.g. for debugging).
#' @param compute_se logical (default TRUE). If TRUE, also computes
#'   `se`/`tval`/`pval` via a numerical Hessian -- an extra O(n^2)
#'   evaluations of the objective on top of the estimation itself. Set
#'   to FALSE to skip this and get only point estimates back faster
#'   (`se`/`tval`/`pval` will be `NULL`).
#'
#' @return list with `param` (`c(margin params, cpar1)` at the optimum,
#'   raw/pre-link2parM scale), `coef` (`margin` converted to natural
#'   units via `link2parM()`, and `cpar1` -- already natural units, no
#'   conversion needed since this function optimizes cpar1 directly, not
#'   the joint model's linked "param" scale), `se`/`tval`/`pval` (same
#'   shape as `coef`, standard errors from a numerical Hessian of the
#'   negative log-likelihood with the margin part delta-method-converted
#'   to match coef's natural units; `tval` is an asymptotic Wald
#'   statistic, not a finite-sample Student-t, so `pval` uses the normal
#'   approximation), `loglik`, `gradient` (shaped like `coef`: a list
#'   with `margin` and `cpar1`, on the RAW optimization scale -- a
#'   convergence diagnostic, not meant to pair with `coef`'s units the
#'   way `se`/`tval`/`pval` do), `mle` (the raw `nlm()` object), and
#'   `used_c_kernel` (whether the compiled kernel was actually used).
#'   NOTE: the `se`/`tval`/`pval` fallback path (for objectives without
#'   an analytic gradient attribute) calls `.central_grad()`, defined in
#'   `EstBifactorCopulaGAMMFast.R` -- source that file too if calling
#'   `EstMargin1Factor()` on its own.
#' @export
EstMargin1Factor <- function(z, clu, margin, cop1, rot1 = 0, dfM = 0, dfC1 = 0,
                              nq = 25, start, LB, UB,
                              kernel_path = "margin1factor_kernel.c",
                              compute_se = TRUE) {

  dismod  <- c("binomial", "bernoulli", "poisson", "nbinom", "geometric", "multinomial")
  is_disc <- margin %in% dismod

  coplik_families <- c("gaussian", "normal", "t", "clayton", "frank",
                        "gumbel", "joe", "plackett", "fgm")
  has_analytic_cop <- cop1 %in% coplik_families

  ## coplik()'s underlying f*ders() functions (fnorders, ftders, fmtcjders,
  ## fgumders, ffrkders, ffgmders, fjoeders, fpladers) now all return the
  ## same column order:
  ##   [C(u|v), dC(u|v)/dcpar, c(u,v), dc(u,v)/dcpar, pdfu]
  ## (fjoeders/fpladers were previously the odd ones out, which meant
  ## coplik()'s uniform rot=90/180 code `out[,c(2,5)] = -out[,c(2,5)]` was
  ## silently negating the copula density itself for joe/plackett -- now
  ## fixed at the source by making every family's column order consistent,
  ## so that sign-flip code is correct for all of them.) This wrapper just
  ## reorders into the canonical layout this file uses everywhere below.
  .coplik_canon <- function(u, v, family, rot, cpar, dfC, du) {
    fam <- if (family == "normal") "gaussian" else family
    raw <- coplik(u, v, fam, rot, cpar, dfC, du)
    # raw = [ccond, ccond1, pdf, pdf1, pdfu] -> canonical order below
    raw[, c(1, 3, 2, 4, 5), drop = FALSE]
  }
  # Canonical column order used everywhere below:
  #   [1] C(u|v)   [2] c(u,v) (density)   [3] dC(u|v)/dcpar
  #   [4] dc(u,v)/dcpar                   [5] pdfu (dc/du if du=TRUE)

  if (is.factor(clu)) clusters <- levels(clu) else clusters <- unique(clu)
  K <- length(clusters)
  ord_list <- lapply(seq_len(K), function(k) which(clu == clusters[k]))
  nn <- vapply(ord_list, length, integer(1))

  gl <- statmod::gauss.quad.prob(nq)
  nl <- gl$nodes
  wl <- gl$weights

  nth <- length(start) - 1L   # number of margin parameters (1 or 2)

  ## ---- 0-based cluster grouping + family code, for the C kernel ----
  order0 <- as.integer(unlist(ord_list) - 1L)
  cbound <- as.integer(c(0, cumsum(nn)))
  K_i <- as.integer(K); nq_i <- as.integer(nq); N_i <- as.integer(length(z))
  nl_d <- as.double(nl); wl_d <- as.double(wl)

  family_code <- function(fam) {
    fam <- ifelse(fam == "normal", "gaussian", fam)
    codes <- c(gaussian = 0L, t = 1L, clayton = 2L, frank = 3L, gumbel = 4L,
               joe = 5L, plackett = 6L, galambos = 7L, `huesler-reiss` = 8L,
               fgm = 9L, ind = 10L)
    unname(codes[fam])
  }
  fam1_code <- family_code(cop1)   # NA if not in coplik.R's coverage

  use_c <- FALSE
  if (!is.null(kernel_path) && !is.na(fam1_code)) {
    use_c <- tryCatch({
      .load_margin1factor_kernel(kernel_path)
      TRUE
    }, error = function(e) {
      message("Note: could not load/compile the Step-1 C kernel (",
              conditionMessage(e), "). Falling back to the pure-R evaluator.")
      FALSE
    })
  }

  ## ---- fully analytic path: continuous margin + coplik-covered family ----
  # Defensive guard shared by every objective below: converts any
  # accidental NaN/Inf (e.g. an extreme but technically in-bounds cpar1
  # producing a numerical singularity inside coplik()/dcop()/pcond())
  # into the same finite `1e10` penalty already used for genuine
  # infeasibility, with a zero gradient of matching length attached (nlm
  # expects a "gradient" attribute on every call once it's used any).
  .finite_penalty <- function(total, grad) {
    # NOTE (fix): the finite branch previously called `.finite_penalty(total,
    # grad)` again instead of returning -- since `total` is still finite on
    # that recursive call, it recurses into itself unconditionally and
    # instantly on every ordinary (non-broken) likelihood evaluation, which
    # is exactly R's "evaluation nested too deeply: infinite recursion" /
    # "node stack overflow" (the byte-compiled form of the same error).
    # It also meant the real, computed `grad` was never actually attached
    # as the "gradient" attribute nlm() needs in the normal case -- only
    # the all-zero placeholder in the infeasible branch ever got attached.
    # Fixed to terminate and attach the real gradient in the finite case.
    if (!is.finite(total)) {
      total <- 1e10
      attr(total, "gradient") <- rep(0, length(grad))
      return(total)
    }
    attr(total, "gradient") <- grad
    total
  }

  negloglik_analytic <- function(param) {
    if (any(param < LB) || any(param > UB)) {
      out <- 1e10
      attr(out, "gradient") <- rep(0, length(param))
      return(out)
    }
    th    <- param[1:nth]
    cpar1 <- param[nth + 1]

    m  <- margins(z, th, margin, dfM = dfM)   # cbind(cdf,pdf,cdf1,cdf2,pdf1,pdf2)
    Gy <- m[, 1]; gy <- m[, 2]
    dG <- m[, c(3, 4)[seq_len(nth)], drop = FALSE]  # dG/dth_p
    dg <- m[, c(5, 6)[seq_len(nth)], drop = FALSE]  # dg/dth_p

    total <- 0
    grad  <- rep(0, nth + 1)

    for (k in seq_len(K)) {
      idx <- ord_list[[k]]
      n_k <- nn[k]
      u1  <- Gy[idx]

      uu <- rep(u1, each = nq)
      vv <- rep(nl, n_k)
      cp <- .coplik_canon(uu, vv, cop1, rot1, cpar1, dfC1, du = TRUE)
      # cp columns: [1]=C(u|v) [2]=c(u,v) [3]=dC(u|v)/dcpar1 [4]=dc(u,v)/dcpar1 [5]=dc(u,v)/du
      logh <- log(rep(gy[idx], each = nq)) + log(pmax(cp[, 2], 1e-300))

      dlogh_th <- matrix(0, n_k * nq, nth)
      for (p in seq_len(nth)) {
        dlogh_th[, p] <- rep(dg[idx, p] / gy[idx], each = nq) +
          (cp[, 5] / cp[, 2]) * rep(dG[idx, p], each = nq)
      }
      dlogh_cpar1 <- cp[, 4] / cp[, 2]

      logh_mat <- matrix(logh, ncol = nq, byrow = TRUE)
      logH_q <- colSums(logh_mat)          # length nq: log H_k(q) = sum_i log h_i(q)
      lmax <- max(logH_q)
      wq <- wl * exp(logH_q - lmax)
      Lk <- sum(wq)
      total <- total - log(Lk) - lmax

      post_w <- wq / Lk                    # posterior weight per quadrature node

      for (p in seq_len(nth)) {
        dsum_q <- colSums(matrix(dlogh_th[, p], ncol = nq, byrow = TRUE))
        grad[p] <- grad[p] - sum(post_w * dsum_q)
      }
      dcpar1_q <- colSums(matrix(dlogh_cpar1, ncol = nq, byrow = TRUE))
      grad[nth + 1] <- grad[nth + 1] - sum(post_w * dcpar1_q)
    }

    .finite_penalty(total, grad)
  }

  ## ---- discrete margin + coplik-covered family: analytic for cpar1,
  ##      central difference (on the margin parameters only) for the rest ----
  negloglik_mixed <- function(param) {
    if (any(param < LB) || any(param > UB)) {
      out <- 1e10
      attr(out, "gradient") <- rep(0, length(param))
      return(out)
    }
    th    <- param[1:nth]
    cpar1 <- param[nth + 1]

    eval_logh <- function(th_) {
      m  <- margins(z, th_, margin, dfM = dfM)
      Gy <- m[, 1]; gy <- m[, 2]
      list(Gy = Gy, gy = gy)
    }
    mm <- eval_logh(th)
    u1 <- mm$Gy
    u2 <- mm$Gy - mm$gy   # G(y-) = G(y) - P(Y=y), no clamping (matches F2<-F1-f elsewhere)

    total <- 0
    grad  <- rep(0, nth + 1)
    heps  <- 1e-5

    for (k in seq_len(K)) {
      idx <- ord_list[[k]]
      n_k <- nn[k]
      uu1 <- rep(u1[idx], each = nq)
      uu2 <- rep(u2[idx], each = nq)
      vv  <- rep(nl, n_k)

      cp1 <- .coplik_canon(uu1, vv, cop1, rot1, cpar1, dfC1, du = FALSE)
      cp2 <- .coplik_canon(uu2, vv, cop1, rot1, cpar1, dfC1, du = FALSE)
      hval <- pmax(cp1[, 1] - cp2[, 1], 1e-100)
      logh <- log(hval)
      dlogh_cpar1 <- (cp1[, 3] - cp2[, 3]) / hval

      logh_mat <- matrix(logh, ncol = nq, byrow = TRUE)
      logH_q <- colSums(logh_mat)
      lmax <- max(logH_q)
      wq <- wl * exp(logH_q - lmax)
      Lk <- sum(wq)
      total <- total - log(Lk) - lmax
      post_w <- wq / Lk

      dcpar1_q <- colSums(matrix(dlogh_cpar1, ncol = nq, byrow = TRUE))
      grad[nth + 1] <- grad[nth + 1] - sum(post_w * dcpar1_q)

      # central difference on the margin parameters (cheap: nth is 1-2)
      for (p in seq_len(nth)) {
        thp <- th; thp[p] <- th[p] + heps
        thm <- th; thm[p] <- th[p] - heps
        mp <- margins(z[idx], thp, margin, dfM = dfM)
        mmv <- margins(z[idx], thm, margin, dfM = dfM)
        u1p <- mp[, 1]; u2p <- mp[, 1] - mp[, 2]
        u1m <- mmv[, 1]; u2m <- mmv[, 1] - mmv[, 2]
        cp1p <- .coplik_canon(rep(u1p, each = nq), vv, cop1, rot1, cpar1, dfC1, du = FALSE)
        cp2p <- .coplik_canon(rep(u2p, each = nq), vv, cop1, rot1, cpar1, dfC1, du = FALSE)
        cp1m <- .coplik_canon(rep(u1m, each = nq), vv, cop1, rot1, cpar1, dfC1, du = FALSE)
        cp2m <- .coplik_canon(rep(u2m, each = nq), vv, cop1, rot1, cpar1, dfC1, du = FALSE)
        hp <- pmax(cp1p[, 1] - cp2p[, 1], 1e-100)
        hm <- pmax(cp1m[, 1] - cp2m[, 1], 1e-100)
        logHp <- colSums(matrix(log(hp), ncol = nq, byrow = TRUE))
        logHm <- colSums(matrix(log(hm), ncol = nq, byrow = TRUE))
        # d/dth_p of log Lk via the same posterior-weighted average, at
        # first order this reduces to a simple weighted central difference
        dlogLk <- sum(post_w * (logHp - logHm)) / (2 * heps)
        grad[p] <- grad[p] - dlogLk
      }
    }

    .finite_penalty(total, grad)
  }

  ## ---- fully analytic path, C kernel (continuous margin, coplik-covered family) ----
  negloglik_analytic_c <- function(param) {
    if (any(param < LB) || any(param > UB)) {
      out <- 1e10
      attr(out, "gradient") <- rep(0, length(param))
      return(out)
    }
    th    <- param[1:nth]
    cpar1 <- param[nth + 1]

    m  <- margins(z, th, margin, dfM = dfM)
    Gy <- m[, 1]; gy <- m[, 2]
    dG2v <- if (nth > 1) m[, 4] else rep(0, N_i)
    dg2v <- if (nth > 1) m[, 6] else rep(0, N_i)

    res <- .C("margin1factor_negloglik_grad",
              Gy = as.double(Gy), gy = as.double(gy),
              dG1 = as.double(m[, 3]), dG2 = as.double(dG2v),
              dg1 = as.double(m[, 5]), dg2 = as.double(dg2v),
              N = N_i, nth = as.integer(nth), cont = 1L, want_th_grad = 1L,
              fam1 = as.integer(fam1_code), rot1 = as.integer(rot1),
              cpar1 = as.double(cpar1), dfC1 = as.double(dfC1),
              order0 = order0, cbound = cbound, K = K_i,
              nl = nl_d, wl = wl_d, nq = nq_i,
              out = double(1), grad_th = double(2), grad_cpar1 = double(1))

    total <- res$out
    attr(total, "gradient") <- c(res$grad_th[seq_len(nth)], res$grad_cpar1)
    total
  }

  ## ---- discrete margin, coplik-covered family, C kernel: analytic
  ##      cpar1 gradient from the kernel, margin-parameter gradient via
  ##      central difference of the kernel's (now fast) total output ----
  negloglik_mixed_c <- function(param) {
    if (any(param < LB) || any(param > UB)) {
      out <- 1e10
      attr(out, "gradient") <- rep(0, length(param))
      return(out)
    }
    th    <- param[1:nth]
    cpar1 <- param[nth + 1]
    dummy <- rep(0, N_i)

    call_kernel <- function(Gy_, gy_, cpar1_) {
      .C("margin1factor_negloglik_grad",
         Gy = as.double(Gy_), gy = as.double(gy_),
         dG1 = dummy, dG2 = dummy, dg1 = dummy, dg2 = dummy,
         N = N_i, nth = as.integer(nth), cont = 0L, want_th_grad = 0L,
         fam1 = as.integer(fam1_code), rot1 = as.integer(rot1),
         cpar1 = as.double(cpar1_), dfC1 = as.double(dfC1),
         order0 = order0, cbound = cbound, K = K_i,
         nl = nl_d, wl = wl_d, nq = nq_i,
         out = double(1), grad_th = double(2), grad_cpar1 = double(1))
    }

    m <- margins(z, th, margin, dfM = dfM)
    base <- call_kernel(m[, 1], m[, 2], cpar1)
    total <- base$out
    grad <- rep(0, nth + 1)
    grad[nth + 1] <- base$grad_cpar1

    heps <- 1e-5
    for (p in seq_len(nth)) {
      thp <- th; thp[p] <- th[p] + heps
      thm <- th; thm[p] <- th[p] - heps
      mp <- margins(z, thp, margin, dfM = dfM)
      mmv <- margins(z, thm, margin, dfM = dfM)
      outp <- call_kernel(mp[, 1], mp[, 2], cpar1)$out
      outm <- call_kernel(mmv[, 1], mmv[, 2], cpar1)$out
      grad[p] <- (outp - outm) / (2 * heps)
    }

    .finite_penalty(total, grad)
  }

  ## ---- fully numeric fallback (any family not covered by coplik()) ----
  negloglik_numeric <- function(param) {
    if (any(param < LB) || any(param > UB)) return(1e10)
    th    <- param[1:nth]
    cpar1 <- param[nth + 1]

    out0 <- margins0(z, th, margin, dfM = dfM)
    F1 <- out0[, 1]; f <- out0[, 2]
    if (is_disc) F2 <- F1 - f

    total <- 0
    for (k in seq_len(K)) {
      idx <- ord_list[[k]]
      n_k <- nn[k]
      vv <- rep(nl, n_k)
      if (!is_disc) {
        u1 <- rep(F1[idx], each = nq)
        ff <- rep(f[idx], each = nq)
        logh <- log(ff) + log(pmax(dcop(u1, vv, cop1, rot1, cpar1, dfC1), 1e-300))
      } else {
        u1 <- rep(F1[idx], each = nq)
        u2 <- rep(F2[idx], each = nq)
        hval <- pmax(pcond(u1, vv, cop1, rot1, cpar1, dfC1) -
                       pcond(u2, vv, cop1, rot1, cpar1, dfC1), 1e-100)
        logh <- log(hval)
      }
      logH_q <- colSums(matrix(logh, ncol = nq, byrow = TRUE))
      lmax <- max(logH_q)
      Lk <- sum(wl * exp(logH_q - lmax))
      total <- total - log(Lk) - lmax
    }
    if (!is.finite(total)) return(1e10)
    total
  }

  # NOTE (fix): `objective` was previously only assigned inside the
  # `if (has_analytic_cop)` branch, left undefined in the `else` branch
  # -- harmless before (nlm() was called with negloglik_numeric directly
  # in that branch), but now needed afterward (for the Hessian/SE code
  # below) regardless of which branch was taken, so it's assigned in
  # both unconditionally.
  if (has_analytic_cop) {
    objective <- if (is_disc) {
      if (use_c) negloglik_mixed_c else negloglik_mixed
    } else {
      if (use_c) negloglik_analytic_c else negloglik_analytic
    }
  } else {
    objective <- negloglik_numeric
  }
  mle <- nlm(objective, p = start, check.analyticals = FALSE, print.level = 0)

  # coef: split mle$estimate into its two natural pieces -- this margin's
  # own parameter(s), converted to natural units via link2parM() (same
  # convention as EstBifactorCopulaGAMMFast()/EstBifactorTwoStep()'s
  # coef$margins -- e.g. Poisson: exp(theta) -> the actual rate), and
  # the copula-1 parameter, which needs NO such conversion: unlike the
  # joint model's copula parameters (which go through linkCop() from an
  # unconstrained "param"), EstMargin1Factor optimizes cpar1 directly in
  # its own natural units already (coplik()/dcop()/pcond() are called
  # with cpar1 as-is, no link layer here).
  theta_hat <- mle$estimate[seq_len(nth)]
  cpar1_hat <- mle$estimate[nth + 1]
  coef <- list(margin = link2parM(theta_hat, margin)[seq_len(nth)], cpar1 = cpar1_hat)
  # gradient: same split as coef, rather than one flat vector -- matching
  # the gradient$copula/gradient$margin convention used by
  # EstBifactorCopulaGAMM(Fast)()/EstBifactorTwoStep(). Unlike `coef`,
  # this stays on the RAW (pre-link2parM) scale EstMargin1Factor actually
  # optimizes over -- the usual convention (a gradient is a convergence
  # diagnostic in the optimizer's own coordinates; se/tval/pval below are
  # what pair with the natural-units coef).
  gradient <- list(margin = mle$gradient[seq_len(nth)], cpar1 = mle$gradient[nth + 1])

  # ---- standard errors / t-values / p-values (optional: this Hessian
  # is O(n^2) evaluations of the objective, on top of the estimation
  # itself -- skip it with compute_se=FALSE if you only need point
  # estimates, or if you're debugging something else entirely) ----
  if (compute_se) {
    # Hessian of the negative log-likelihood w.r.t. the RAW optimization
    # variables, via central differences of a gradient function (reusing
    # whichever objective was actually used; its analytic `gradient`
    # attribute is used directly when present, else a numeric fallback).
    grad_full <- function(param) {
      val <- objective(param)
      g <- attr(val, "gradient")
      if (is.null(g)) g <- .central_grad(objective, param, LB, UB)
      g
    }
    H <- .numerical_hessian_from_grad(grad_full, mle$estimate, LB, UB)
    V <- tryCatch(solve(H), error = function(e) matrix(NA_real_, nrow(H), ncol(H)))
    d <- diag(V)
    d[!is.na(d) & d < 0] <- NA_real_
    se_raw <- sqrt(d)

    # Delta method: se_raw is on the raw (pre-link) scale; coef is on the
    # natural-units scale, so se needs the same conversion, via the
    # derivative of the relevant link. `.margin_link_deriv()` gets this
    # numerically from link2parM() itself (safer than re-deriving each
    # family's link formula by hand, and automatically stays correct if
    # link2parM() ever changes). cpar1 needs no conversion (see coef above).
    dh_margin <- .margin_link_deriv(theta_hat, margin, nth)
    se_margin <- se_raw[seq_len(nth)] * abs(dh_margin[seq_len(nth)])
    se_cpar1 <- se_raw[nth + 1]
    se <- list(margin = se_margin, cpar1 = se_cpar1)

    # tval: an asymptotic Wald statistic (coef/se), reported as "t" per
    # convention here -- not a Student-t with a meaningful finite-sample
    # df, so pval uses the normal approximation (2*pnorm(-abs(t))), which
    # is what the asymptotic theory underlying this Wald construction
    # actually justifies.
    tval <- list(margin = coef$margin / se$margin, cpar1 = coef$cpar1 / se$cpar1)
    pval <- list(margin = 2 * pnorm(-abs(tval$margin)), cpar1 = 2 * pnorm(-abs(tval$cpar1)))
  } else {
    se <- NULL; tval <- NULL; pval <- NULL
  }

  list(param = mle$estimate, coef = coef, se = se, tval = tval, pval = pval,
       loglik = -mle$minimum, gradient = gradient, mle = mle, used_c_kernel = use_c)
}

#' @title (internal) Numerical Hessian from a gradient function
#' @description Forms the Hessian of a scalar function at `x` via central
#' differences of its gradient (rather than second differences of the
#' function value directly, which is noisier) -- `H[,i]` is
#' `(grad_fn(x + h*e_i) - grad_fn(x - h*e_i)) / (2h)`, symmetrized
#' afterward to cancel small numerical asymmetries.
#' @keywords internal
.numerical_hessian_from_grad <- function(grad_fn, x, lower, upper, h0 = 1e-4) {
  n <- length(x)
  H <- matrix(0, n, n)
  for (i in seq_len(n)) {
    room <- min(x[i] - lower[i], upper[i] - x[i])
    hi <- max(min(h0, 0.4 * room), 1e-6)
    xp <- x; xp[i] <- x[i] + hi
    xm <- x; xm[i] <- x[i] - hi
    H[, i] <- (grad_fn(xp) - grad_fn(xm)) / (2 * hi)
  }
  (H + t(H)) / 2
}

#' @title (internal) Numeric derivative of link2parM()'s natural-units map
#' @description Returns d(natural units)/d(raw theta) elementwise, via
#' central difference of `link2parM()` itself -- used by the delta method
#' to convert a raw-scale standard error into a natural-units one. Only
#' the first `nth` entries are meaningful (matching how many real
#' parameters this margin actually has).
#' @keywords internal
.margin_link_deriv <- function(theta_raw, marg, nth, h = 1e-6) {
  d <- rep(NA_real_, length(theta_raw))
  for (p in seq_len(nth)) {
    tp <- theta_raw; tp[p] <- tp[p] + h
    tm <- theta_raw; tm[p] <- tm[p] - h
    d[p] <- (link2parM(tp, marg)[p] - link2parM(tm, marg)[p]) / (2 * h)
  }
  d
}

#' @title (internal) Load or compile the Step-1 margin/copula-1 C kernel
#' @description Same pattern as `EstBifactorCopulaGAMMFast()`'s
#' `.load_bifactor_kernel()`: loads `margin1factor_kernel.so`/`.dll` if
#' already built and up to date, otherwise compiles
#' `margin1factor_kernel.c` with `R CMD SHLIB`, trying `-fopenmp` first
#' (clusters are independent, same as the joint model's kernel) and
#' falling back to a single-threaded build if that fails.
#' @keywords internal
.load_margin1factor_kernel <- function(kernel_path) {
  if (is.loaded("margin1factor_negloglik_grad")) return(invisible(TRUE))
  if (!file.exists(kernel_path)) stop("kernel_path not found: ", kernel_path)

  if (grepl("\\.c$", kernel_path)) {
    dir  <- normalizePath(dirname(kernel_path), mustWork = TRUE)
    file <- basename(kernel_path)
    so_name <- paste0(tools::file_path_sans_ext(file), .Platform$dynlib.ext)
    so_path <- file.path(dir, so_name)

    need_compile <- !file.exists(so_path) ||
      file.mtime(file.path(dir, file)) > file.mtime(so_path)

    if (need_compile) {
      wd <- getwd(); setwd(dir); on.exit(setwd(wd), add = TRUE)

      build_with <- function(cflags, libs) {
        old_cflags <- Sys.getenv("PKG_CFLAGS", unset = NA)
        old_libs   <- Sys.getenv("PKG_LIBS", unset = NA)
        on.exit({
          if (is.na(old_cflags)) Sys.unsetenv("PKG_CFLAGS") else Sys.setenv(PKG_CFLAGS = old_cflags)
          if (is.na(old_libs))   Sys.unsetenv("PKG_LIBS")   else Sys.setenv(PKG_LIBS = old_libs)
        }, add = TRUE)
        Sys.setenv(PKG_CFLAGS = cflags, PKG_LIBS = libs)
        if (file.exists(so_name)) file.remove(so_name)
        system2(file.path(R.home("bin"), "R"), c("CMD", "SHLIB", shQuote(file)),
                stdout = TRUE, stderr = TRUE)
      }

      use_openmp <- isTRUE(getOption("bifactor.use_openmp", TRUE))
      res <- if (use_openmp) build_with("-fopenmp -O2", "-fopenmp") else character(0)
      if (use_openmp && !file.exists(so_name)) {
        message("Note: compiling margin1factor_kernel.c with OpenMP failed; ",
                "retrying a single-threaded build.")
        res <- build_with("-O2", "")
      }
      if (!file.exists(so_name)) stop("R CMD SHLIB failed:\n", paste(res, collapse = "\n"))
    }
    dyn.load(so_path)
  } else {
    dyn.load(kernel_path)
  }
  invisible(TRUE)
}
