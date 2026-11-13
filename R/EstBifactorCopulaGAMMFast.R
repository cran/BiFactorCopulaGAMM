#' @title Bi-factor copula estimation for clustered mixed data (fast version)
#'
#' @description This is a faster replacement for `EstBifactorCopulaGAMM()`.
#' It is NOT numerically identical to the original: the continuous-margin
#' branch of the likelihood has been corrected to match eq. (2.8) of the
#' bi-factor copula paper (see the note below), so expect the estimates to
#' differ from `EstBifactorCopulaGAMM()` whenever the model includes a
#' continuous margin. Everything else about the objective function is
#' unchanged; the only remaining differences are *how* it is evaluated.
#'
#' Why the original is slow: `nlm()` evaluates the objective function many
#' times (it estimates the gradient numerically, so every iteration costs
#' roughly `npar+1` evaluations of `likf()`, and `iterlim=200` allows many
#' iterations). Each call to the original `likf()` loops over every cluster
#' in R and, for every cluster, calls `dcop()`/`pcond()` -- which each
#' re-dispatch through a `switch()` on the family name -- on freshly
#' allocated vectors. That per-call R-interpreter overhead, multiplied by
#' (number of clusters) x (number of `nlm` evaluations), is what dominates
#' the runtime; the actual arithmetic is tiny.
#'
#' This version:
#' \enumerate{
#'   \item Precomputes, once (outside the objective function), everything
#'         that does not change across `nlm` evaluations: the cluster
#'         grouping (row order + boundaries), and the integer family/
#'         rotation/df codes for every copula slot.
#'   \item Moves the per-cluster double-Gauss-quadrature loop (previously
#'         re-run, in R, on every objective evaluation) into a single call
#'         to a compiled C routine, `bifactor_negloglik()`, defined in
#'         `bifactor_kernel.c`. That C routine reproduces the arithmetic of
#'         `dcop()`/`pcond()` exactly (including the `pcond.R` "fgm"
#'         rotation case), with one intentional correctness fix: the
#'         continuous-margin branch of the original `likf()` computed
#'         `temp1 = matff[,j]*dcop(u1,vv,...)` -- the density prefactor
#'         `f_j(x_j)*c_[j,1][F_j(x_j),v]` required by eq. (2.8) of the
#'         bi-factor copula paper -- but then discarded it and multiplied by
#'         the conditional cdf `temp2` in its place. Both this C kernel and
#'         the pure-R fallback below now correctly use `temp1`/`f_j*c_[j,1]`
#'         as the multiplier instead.
#'   \item If the C library cannot be found/compiled/loaded, this function
#'         automatically falls back to a pure-R evaluator (still with the
#'         precomputed cluster grouping, which is a safe, free win on its
#'         own) and prints a one-time message explaining how to compile the
#'         C kernel for the full speedup.
#'   \item Parallelizes the C kernel's cluster loop with OpenMP (clusters
#'         are statistically independent, so this is embarrassingly
#'         parallel and exact -- not an approximation). This is the single
#'         biggest remaining lever if estimation is still slow: with
#'         `method="nlminb"`, the numerical gradient costs `2*length(start)`
#'         evaluations of the C kernel per outer iteration, so on a machine
#'         with several cores this can give a near-linear additional
#'         speedup on top of the C rewrite itself. Compiled automatically
#'         by `.load_bifactor_kernel()`; falls back to a single-threaded
#'         build if the compiler doesn't support `-fopenmp`.
#' }
#'
#' @param Y     matrix of responses (assumed continuous or discrete).
#' @param X     matrix of covariates (assumed continuous or discrete).
#' @param clu   variable of size n defining the clusters; can be a factor
#' @param model list of lists for columns vectors of x. For any j, model[[j]]$margin is a list containing the margin family, param (vector of parameters), and size if necessary.
#' @param start list of vector of starting point for the estimation.
#' @param LB    list of vector of lower bound for the parameters.
#' @param UB    list of vector of upper bound for the parameters.
#' @param nq    number of nodes and weights for Gaussian quadrature of the product of conditional copulas; default is 25.
#' @param prediction logical variable for prediction of latent variables V (default is TRUE).
#' @param kernel_path Path to the compiled shared library (`.so`/`.dll`) or
#'   to `bifactor_kernel.c` (in which case it will be compiled with
#'   `R CMD SHLIB` the first time). Defaults to `"bifactor_kernel.c"` in the
#'   current working directory.
#' @param method Optimizer to use. `"nlminb"` (default) enforces `LB`/`UB`
#'   natively and uses a bound-aware numerical gradient -- recommended, and
#'   the fix for the "numerical gradients are very large / doesn't converge"
#'   failure mode that can occur with `"nlm"` (kept only for comparison /
#'   backward compatibility), which is unconstrained and can have its
#'   finite-difference gradient corrupted whenever a probe step crosses a
#'   tight bound and hits the internal `1e100` infeasibility penalty.
#' @param compute_se logical (default TRUE). If TRUE, also computes
#'   `se`/`tval`/`pval` via a numerical Hessian -- an extra O(npar^2)
#'   evaluations of `likf()` on top of the estimation itself. Set to
#'   FALSE to skip this and get only point estimates back faster
#'   (`se`/`tval`/`pval` will be `NULL`).
#' @return list with `coef` (list with `copulas`/`margins` matrices, same
#'   shape as `EstBifactorTwoStep()`'s `coef` -- `margins` is in
#'   natural units via `link2parM()`, matching that function's own
#'   convention, not the raw pre-link scale Step 1 optimizes over) and
#'   `gradient` (shaped like `coef`: a list with `copula` and `margin`
#'   matrices, each `dp x 2` with the same row order as `coef$copulas`/
#'   `coef$margins`, rather than a flat vector. Same parameter scale as
#'   `EstBifactorTwoStep()`'s `gradient` -- Step 1's copula-1
#'   gradient is chain-rule-converted from the raw copula-parameter
#'   scale it's actually computed on), plus `se`/`tval`/`pval` (same
#'   `list(copula=,margin=)` shape as `coef`, standard errors and
#'   asymptotic Wald tests -- see `se_note` for whether these are the
#'   naive two-step version or the full-MLE version), `se_note` (a
#'   short string explaining which), `loglik`, `aic`, `bic`. These all
#'   come from `final` (the refined joint fit) when `refine=TRUE`;
#'   otherwise they're assembled directly from the Step 1 + Step 2
#'   results -- in that case `se`/`tval`/`pval` are the NAIVE two-step
#'   version (Step 1's margin/cpar1 standard errors reused from each
#'   `EstMargin1Factor()` fit in isolation; Step 2's cpar2 standard
#'   error treats margin/cpar1 as fixed/known), which understate the
#'   true uncertainty -- use `refine=TRUE` for valid inference. Also
#'   returns `gradient_copulas`, a `dp x 2` matrix (same
#'   row order as `coef$copulas`) with columns `"step1"` (the
#'   chain-rule-converted cpar1 gradient from `EstMargin1Factor()`) and
#'   `"step2"` (the cpar2 gradient from the Step 2 fit) -- always the
#'   two-step breakdown, regardless of `refine`, since that distinction
#'   doesn't exist once/if the joint refinement runs. Also returns
#'   `step1` (per-margin `EstMargin1Factor()` results), `step2` (the
#'   copula-2-only fit), `warm_start` (the assembled full parameter
#'   vector -- also used to warm-start the refinement step when
#'   `refine=TRUE`), `final` (the refined joint fit if `refine=TRUE`,
#'   else `NULL`), and `calc_time` (a named vector with the wall-clock
#'   time for each phase).
#' @export
EstBifactorCopulaGAMMFast <- function(Y, X = NULL, clu, model, start = NULL,
                                       LB = NULL, UB = NULL, nq = 25,
                                       prediction = TRUE,
                                       kernel_path = "bifactor_kernel.c",
                                       method = c("nlminb", "nlm"),
                                       compute_se = TRUE) {

  dismod  <- c("binomial", "bernoulli", "poisson", "nbinom", "geometric", "multinomial")
  contmod <- c("gaussian", "normal", "t", "laplace", "exponential", "weibull")
  if (is.null(dim(Y))) { Y <- matrix(Y, ncol = 1) }

  dimY <- dim(Y)
  modY <- model$Y
  d <- length(modY)
  if (d != dimY[2]) stop("Error: wrong dimension for the model of responses")

  if (is.null(X)) {
    dp <- d
  } else {
    if (is.null(dim(X))) { X <- matrix(X, ncol = 1) }
    dimX <- dim(X)
    modX <- model$X
    p <- length(modX)
    if (p != dimX[2]) stop("Error:wrong dimension for the model of covariates")
    dp <- p + d
  }

  if (is.factor(clu)) { clusters <- levels(clu) } else { clusters <- unique(clu) }
  K <- length(clusters)
  N <- dimY[1]

  nn <- rep(0, K)
  for (k in 1:K) {
    ind <- (clu == clusters[k])
    nn[k] <- sum(ind)
  }

  cont  <- rep(1, dp)
  rot   <- matrix(0, ncol = 2, nrow = dp)
  margin <- NULL
  cop   <- NULL
  dfM   <- rep(0, dp)
  dfC   <- matrix(0, ncol = 2, nrow = dp)
  cpar  <- matrix(0, ncol = 2, nrow = dp)

  for (j in 1:d) {
    mar <- model$Y[[j]]$margin$family
    margin <- c(margin, mar)
    df <- model$Y[[j]]$margin$dfM
    if (is.null(df) == FALSE) dfM[j] <- df
    if (mar %in% dismod) cont[j] <- 0
    mod <- model$Y[[j]]
    cop <- c(cop, mod$copula[[1]]$family, mod$copula[[2]]$family)
    df <- mod$copula[[1]]$dfC
    if (is.null(df == FALSE)) dfC[j, 1] <- df
    df <- mod$copula[[2]]$dfC
    if (is.null(df == FALSE)) dfC[j, 2] <- df
    rot1 <- mod$copula[[1]]$rot
    if (is.null(rot1)) { rot[j, 1] <- 0 } else { rot[j, 1] <- rot1 }
    rot2 <- mod$copula[[2]]$rot
    if (is.null(rot2)) { rot[j, 2] <- 0 } else { rot[j, 2] <- rot2 }
  }
  if (dp == 1) cop[2] <- "ind"

  if (dp > d) {
    for (j in 1:p) {
      jj <- j + d
      margin <- c(margin, model$X[[j]]$margin$family)
      df <- model$X[[j]]$margin$dfM
      if (is.null(df) == FALSE) dfM[jj] <- df
      if (margin[jj] %in% dismod) cont[jj] <- 0
      mod <- model$X[[j]]
      cop <- c(cop, mod$copula[[1]]$family, mod$copula[[2]]$family)
      df <- mod$copula[[1]]$dfC
      if (is.null(df == FALSE)) dfC[j, 1] <- df
      df <- mod$copula[[2]]$dfC
      if (is.null(df == FALSE)) dfC[j, 2] <- df
      rot1 <- mod$copula[[1]]$rot
      if (is.null(rot1)) { rot[jj, 1] <- 0 } else { rot[jj, 1] <- rot1 }
      rot2 <- mod$copula[[2]]$rot
      if (is.null(rot2)) { rot[jj, 2] <- 0 } else { rot[jj, 2] <- rot2 }
    }
  }

  dd <- rep(0, dp)
  for (j in 1:dp) dd[j] <- param.dim(margin[j])

  XX <- as.matrix(cbind(Y, X))

  if (is.null(start) || is.null(LB) || is.null(UB)) {
    outm <- bounds.mar(XX, margin)
    outc <- bounds.cop(cop)
    start <- c(outm$start, outc$start)
    UB <- c(outm$UB, outc$UB)
    LB <- c(outm$LB, outc$LB)
  }

  parM <- matrix(0, ncol = 2, nrow = dp)

  gl <- statmod::gauss.quad.prob(nq)
  nl <- gl$nodes
  wl <- gl$weights

  F1 <- matrix(0, ncol = dp, nrow = N)
  F2 <- F1
  f  <- matrix(1, ncol = dp, nrow = N)

  ## ---- one-time precomputation (does not depend on the optimized params) ----

  # cluster grouping: 0-based row order and cluster boundaries, for the C kernel
  ord <- unlist(lapply(seq_len(K), function(k) which(clu == clusters[k])))
  order0 <- as.integer(ord - 1L)
  cbound <- as.integer(c(0, cumsum(nn)))

  # integer family codes (must match bifactor_kernel.c's family_code mapping)
  family_code <- function(fam) {
    fam <- ifelse(fam == "normal", "gaussian", fam)
    codes <- c(gaussian = 0L, t = 1L, clayton = 2L, frank = 3L, gumbel = 4L,
               joe = 5L, plackett = 6L, galambos = 7L, `huesler-reiss` = 8L,
               fgm = 9L, ind = 10L)
    out <- unname(codes[fam])
    if (any(is.na(out))) stop("Unknown copula family: ", paste(fam[is.na(out)], collapse = ", "))
    as.integer(out)
  }
  fam1 <- family_code(cop[seq(1, 2 * dp, by = 2)])
  fam2 <- family_code(cop[seq(2, 2 * dp, by = 2)])
  rot1v <- as.integer(rot[, 1])
  rot2v <- as.integer(rot[, 2])
  dfC1v <- as.double(dfC[, 1])
  dfC2v <- as.double(dfC[, 2])
  contv <- as.integer(cont)
  nl_d <- as.double(nl)
  wl_d <- as.double(wl)
  N_i <- as.integer(N)
  dp_i <- as.integer(dp)
  K_i <- as.integer(K)
  nq_i <- as.integer(nq)

  ## ---- load / compile the C kernel, with an automatic pure-R fallback ----
  use_c <- FALSE
  if (!is.null(kernel_path)) {
    use_c <- tryCatch({
      .load_bifactor_kernel(kernel_path)
      TRUE
    }, error = function(e) {
      message("Note: could not load/compile the C kernel (", conditionMessage(e),
              "). Falling back to a pure-R evaluator (still faster than the ",
              "original due to precomputed cluster indices, but without the ",
              "full C speedup).")
      FALSE
    })
  }

  #loglikelihood
  likf <- function(param) {

    l <- 0
    for (j in 1:dp) {
      if (dd[j] == 1) { parM[j, 1] <- param[(l + 1)] } else { parM[j, ] <- param[(l + 1):(l + 2)] }
      l <- l + dd[j]
    }
    dm <- l
    parC <- matrix(param[(dm + 1):(dm + 2 * dp)], ncol = 2, byrow = TRUE)
    for (j in 1:dp) {
      cpar[j, 1] <- linkCop(parC[j, 1], cop[(2 * j - 1)])$cpar
      cpar[j, 2] <- linkCop(parC[j, 2], cop[(2 * j)])$cpar
    }

    if (min(param - LB) < 0 || max(param - UB) > 0) return(1e100)

    for (j in 1:dp) {
      out0 <- margins0(XX[, j], parM[j, ], margin[j], dfM = dfM[j])
      F1[, j] <- out0[, 1]
      f[, j]  <- out0[, 2]
      if (cont[j] == 0) {
        F2[, j] <- F1[, j] - f[, j]
        f[, j] <- 1
      }
    }

    if (use_c) {
      res <- .C("bifactor_negloglik",
                F1 = as.double(F1), F2 = as.double(F2), f = as.double(f),
                N = N_i, dp = dp_i,
                cont = contv,
                fam1 = fam1, fam2 = fam2,
                rot1 = rot1v, rot2 = rot2v,
                cpar1 = as.double(cpar[, 1]), cpar2 = as.double(cpar[, 2]),
                dfC1 = dfC1v, dfC2 = dfC2v,
                order0 = order0, cbound = cbound, K = K_i,
                nl = nl_d, wl = wl_d, nq = nq_i,
                out = double(1))
      # Defensive guard: see the note below the pure-R fallback for why.
      if (!is.finite(res$out)) return(1e100)
      return(res$out)
    }

    # ---- pure-R fallback (same math, precomputed cluster indices; the
    #      continuous-margin branch uses the f_j(x_j)*c_[j,1][...] density
    #      prefactor per eq. (2.8), not the conditional cdf) ----
    total <- 0
    for (k in 1:K) {
      rows <- (cbound[k] + 1):cbound[k + 1]
      idx  <- order0[rows] + 1L
      n_k  <- length(idx)

      FF1 <- F1[idx, , drop = FALSE]
      FF2 <- F2[idx, , drop = FALSE]
      ff  <- f[idx, , drop = FALSE]

      FFF1 <- rep(FF1, each = nq)
      FFF2 <- rep(FF2, each = nq)
      fff  <- rep(ff, each = nq)
      vv <- rep(nl, n_k)
      w  <- rep(nl, nq * n_k)
      matFF1 <- matrix(FFF1, ncol = dp)
      matFF2 <- matrix(FFF2, ncol = dp)
      matff  <- matrix(fff, ncol = dp)

      LL <- 1
      for (j in 1:dp) {
        u1 <- matFF1[, j]
        u2 <- matFF2[, j]
        if (cont[j]) {
          dens1 <- matff[, j] * dcop(u1, vv, cop[(2 * j - 1)], rot[j, 1], cpar[j, 1], dfC[j, 1])
          dens1 <- rep(dens1, each = nq)
          temp2 <- pcond(u1, vv, cop[(2 * j - 1)], rot[j, 1], cpar[j, 1], dfC[j, 1])
          temp2 <- rep(temp2, each = nq)
          temp3 <- dens1 * dcop(temp2, w, cop[(2 * j)], rot[j, 2], cpar[j, 2], dfC[j, 2])
          LL <- LL * temp3
        } else {
          temp1 <- pcond(u1, vv, cop[(2 * j - 1)], rot[j, 1], cpar[j, 1], dfC[j, 1])
          temp2 <- pcond(u2, vv, cop[(2 * j - 1)], rot[j, 1], cpar[j, 1], dfC[j, 1])
          temp1 <- rep(temp1, each = nq)
          temp2 <- rep(temp2, each = nq)
          temp3 <- pcond(temp1, w, cop[(2 * j)], rot[j, 2], cpar[j, 2], dfC[j, 2])
          temp4 <- pcond(temp2, w, cop[(2 * j)], rot[j, 2], cpar[j, 2], dfC[j, 2])
          LL <- LL * (temp3 - temp4)
        }
      }
      LL <- matrix(LL, ncol = nq, byrow = TRUE)
      h <- LL %*% wl
      h <- pmax(h, 1e-100)
      h <- matrix(h, ncol = nq, byrow = TRUE)
      ltem3 <- colSums(log(h))
      lmax3 <- max(ltem3)
      wprd <- wl * exp(ltem3 - lmax3)
      intf <- sum(wprd)
      total <- total - log(intf) - lmax3
    }
    # Defensive guard: convert any accidental NaN/Inf (e.g. from an
    # extreme but technically in-bounds copula/margin parameter combo
    # producing a numerical singularity inside dcop()/pcond()) into the
    # same finite `1e100` penalty already used for genuine infeasibility,
    # rather than letting it propagate and trigger nlminb's/nlm's
    # "NA/NaN function evaluation" warning (or worse, corrupt a
    # subsequent finite-difference gradient step).
    if (!is.finite(total)) return(1e100)
    total
  }

  #
  method <- match.arg(method, c("nlminb", "nlm"))
  start_time <- Sys.time()
  if (method == "nlminb") {
    # nlminb() (PORT algorithm) enforces lower/upper natively -- it never
    # evaluates likf() outside [LB,UB], so the `1e100` penalty branch above
    # never fires and can no longer corrupt a numerical gradient the way it
    # does under nlm(). We also supply our own bound-aware central-difference
    # gradient (see .central_grad()), whose step size is automatically
    # shrunk near a bound, instead of letting nlminb pick a default step
    # that might still cross a tight bound (e.g. a width-0.06 SD window).
    grad_fn <- function(param) .central_grad(likf, param, LB, UB)
    fit <- nlminb(start = start, objective = likf, gradient = grad_fn,
                  lower = LB, upper = UB,
                  control = list(iter.max = 200, eval.max = 400))
    mle <- list(estimate = fit$par, minimum = fit$objective,
                gradient = grad_fn(fit$par),
                code = fit$convergence, message = fit$message)
  } else {
    # Original behavior, kept for comparison / backward compatibility.
    # typsize scales nlm's internal finite-difference step per parameter,
    # which helps somewhat with the very different natural scales here
    # (probabilities ~0.4, counts ~1.7, means ~10, SDs ~0.2) but does NOT
    # fix the boundary-crossing problem described above -- nlm has no
    # notion of LB/UB at all.
    mle <- nlm(likf, p = start, check.analyticals = FALSE, print.level = 0,
               iterlim = 200, typsize = pmax(abs(start), 0.1))
  }
  end_time <- Sys.time()
  calc_time <- end_time - start_time
  calc_time

  param <- mle$estimate

  parM0 <- matrix(0, ncol = 2, nrow = dp)
  l <- 0
  for (j in 1:dp) {
    if (dd[j] == 1) { parM0[j, 1] <- param[(l + 1)] } else { parM0[j, ] <- param[(l + 1):(l + 2)] }
    l <- l + dd[j]
  }
  for (j in 1:dp) {
    parM[j, ] <- link2parM(parM0[j, ], margin[j])
  }

  dm <- l
  parC <- matrix(param[(dm + 1):(dm + 2 * dp)], ncol = 2, byrow = TRUE)
  for (j in 1:dp) {
    cpar[j, 1] <- linkCop(parC[j, 1], cop[(2 * j - 1)])$cpar
    cpar[j, 2] <- linkCop(parC[j, 2], cop[(2 * j)])$cpar
  }

  LL <- mle$minimum
  npar <- length(param)
  AIC <- 2 * npar + 2 * LL
  BIC <- npar * log(N) + 2 * LL
  coef <- list(copulas = cpar, margins = parM)
  # Reshape mle$gradient (a flat vector, ordered margin-block then
  # copula-block, same layout as `param`) into the same shape as `coef`:
  # a list with `copula` and `margin` matrices, each dp x 2 with the
  # same row order as coef$copulas/coef$margins -- rather than a flat
  # vector. `margin`'s second column is NA (not 0) whenever dd[j]==1,
  # since there is no second parameter for that margin and so no
  # gradient was ever computed for it.
  dm_off <- c(0, cumsum(dd))
  grad_margin <- matrix(NA_real_, ncol = 2, nrow = dp)
  for (j in 1:dp) grad_margin[j, 1:dd[j]] <- mle$gradient[(dm_off[j] + 1):(dm_off[j] + dd[j])]
  grad_copula <- matrix(mle$gradient[(dm + 1):(dm + 2 * dp)], ncol = 2, byrow = TRUE,
                         dimnames = list(NULL, c("cpar1", "cpar2")))
  gradient <- list(copula = grad_copula, margin = grad_margin)

  # ---- standard errors / t-values / p-values (optional: this Hessian
  # is O(npar^2) evaluations of likf() on top of the estimation itself --
  # skip it with compute_se=FALSE if you only need point estimates) ----
  if (compute_se) {
    # Hessian of the negative log-likelihood w.r.t. the RAW `param` vector
    # (the space likf() is actually optimized over), via central
    # differences of a numerical gradient function (likf() has no
    # analytic gradient attribute -- only Step 1's single-quadrature
    # model, in EstMargin1Factor(), got that treatment -- so .central_grad()
    # is used directly here, same as the nlminb branch's own grad_fn).
    grad_full <- function(p) .central_grad(likf, p, LB, UB)
    H <- .numerical_hessian_from_grad(grad_full, param, LB, UB)
    V <- tryCatch(solve(H), error = function(e) matrix(NA_real_, nrow(H), ncol(H)))
    d_diag <- diag(V)
    d_diag[!is.na(d_diag) & d_diag < 0] <- NA_real_
    se_raw <- sqrt(d_diag)

    # Delta method: se_raw is on the raw `param` scale; coef (cpar, parM)
    # is on natural units (via linkCop()/link2parM()), so se needs the
    # same conversion. Derivatives obtained numerically from linkCop()/
    # link2parM() themselves (safer than re-deriving each family's link
    # formula by hand, and stays correct if those functions change).
    se_margin <- matrix(NA_real_, ncol = 2, nrow = dp)
    se_copula <- matrix(NA_real_, ncol = 2, nrow = dp, dimnames = list(NULL, c("cpar1", "cpar2")))
    for (j in 1:dp) {
      dh <- .margin_link_deriv(parM0[j, ], margin[j], dd[j])
      se_margin[j, 1:dd[j]] <- se_raw[(dm_off[j] + 1):(dm_off[j] + dd[j])] * abs(dh[1:dd[j]])

      se_raw_c1 <- se_raw[dm + 2 * j - 1]
      se_raw_c2 <- se_raw[dm + 2 * j]
      dc1 <- .cop_link_deriv(parC[j, 1], cop[(2 * j - 1)])
      dc2 <- .cop_link_deriv(parC[j, 2], cop[(2 * j)])
      se_copula[j, 1] <- se_raw_c1 * abs(dc1)
      se_copula[j, 2] <- se_raw_c2 * abs(dc2)
    }
    se <- list(copula = se_copula, margin = se_margin)

    # tval: an asymptotic Wald statistic (coef/se), not a finite-sample
    # Student-t, so pval uses the normal approximation.
    tval <- list(copula = coef$copulas / se$copula, margin = coef$margins / se$margin)
    pval <- list(copula = 2 * pnorm(-abs(tval$copula)), margin = 2 * pnorm(-abs(tval$margin)))
  } else {
    se <- NULL; tval <- NULL; pval <- NULL
  }

  out <- list(coef = coef, gradient = gradient, se = se, tval = tval, pval = pval,
              loglik = -LL, aic = AIC, bic = BIC,
              clu = clu, clusters = clusters, margins = margin, copulas = cop, rot = rot,
              model = model, dfC = dfC, dfM = dfM, calc_time = calc_time, used_c_kernel = use_c)
  out
}

#' @title (internal) Bound-aware central-difference gradient
#' @description Computes a numerical gradient of `f` at `x` using central
#' differences, with a per-parameter step size that is automatically
#' shrunk so the probe points `x[i] +/- h[i]` never cross `lower[i]`/
#' `upper[i]`. This is what actually fixes (rather than papers over) the
#' "exploding gradient" failure mode: a plain finite-difference gradient
#' has no notion of a feasible region and will happily probe past a tight
#' bound, hitting whatever infeasibility penalty the objective returns
#' there and producing a wildly inflated derivative estimate for that
#' coordinate.
#' @keywords internal
.central_grad <- function(f, x, lower, upper, h0 = 1e-4) {
  # NOTE (fix): the previous version computed `hi <- max(min(h0, 0.4 *
  # room), 1e-8)` where `room = min(x[i]-lower[i], upper[i]-x[i])`. When
  # a parameter sits at (or is clamped to) a bound -- e.g. from a warm
  # start built by EstBifactorTwoStep()'s invfunc()-based inversion --
  # `room` can be 0 or tiny, but the `1e-8` floor still forced a step of
  # at least that size in BOTH directions, so one of `x[i]+hi`/`x[i]-hi`
  # would land outside [lower[i],upper[i]]. That probe hits likf()'s own
  # `1e100` infeasibility penalty, and `(1e100 - finite)/(2*hi)` becomes
  # an enormous, corrupted gradient component -- which can easily send
  # the NEXT optimizer step somewhere the likelihood genuinely breaks
  # down numerically (matching nlminb's "NA/NaN function evaluation"
  # warning). Fixed by falling back to a ONE-SIDED difference, using
  # only whichever side actually has room, whenever the two-sided step
  # would cross a bound.
  n <- length(x)
  g <- numeric(n)
  for (i in 1:n) {
    room_plus  <- upper[i] - x[i]
    room_minus <- x[i] - lower[i]
    room <- min(room_plus, room_minus)
    if (room < 2e-8) {
      if (room_plus >= room_minus) {
        hi <- max(min(h0, 0.4 * room_plus), 1e-10)
        xp <- x; xp[i] <- x[i] + hi
        g[i] <- (f(xp) - f(x)) / hi
      } else {
        hi <- max(min(h0, 0.4 * room_minus), 1e-10)
        xm <- x; xm[i] <- x[i] - hi
        g[i] <- (f(x) - f(xm)) / hi
      }
    } else {
      hi <- max(min(h0, 0.4 * room), 1e-8)
      xp <- x; xp[i] <- x[i] + hi
      xm <- x; xm[i] <- x[i] - hi
      g[i] <- (f(xp) - f(xm)) / (2 * hi)
    }
  }
  g
}

#' @title (internal) Numerical Hessian from a gradient function
#' @description Forms the Hessian of a scalar function at `x` via central
#' differences of its gradient (rather than second differences of the
#' function value directly, which is noisier) -- `H[,i]` is
#' `(grad_fn(x + h*e_i) - grad_fn(x - h*e_i)) / (2h)`, symmetrized
#' afterward to cancel small numerical asymmetries. Used to get standard
#' errors (via the inverse Hessian) for `se`/`tval`/`pval`.
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
#' central difference of `link2parM()` itself -- used by the delta
#' method to convert a raw-scale standard error into a natural-units
#' one. Only the first `nth` entries are meaningful.
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

#' @title (internal) Numeric derivative of linkCop()'s cpar map
#' @description Returns d(cpar)/d(param), via central difference of
#' `linkCop(param, family)$cpar` -- used by the delta method to convert
#' a raw-scale (pre-link) copula-parameter standard error into the
#' natural-units scale `coef$copulas` is reported in.
#' @keywords internal
.cop_link_deriv <- function(param_raw, family, h = 1e-6) {
  (linkCop(param_raw + h, family)$cpar - linkCop(param_raw - h, family)$cpar) / (2 * h)
}

#' @title (internal) Load or compile the bifactor C kernel
#' @description Loads `bifactor_kernel.so`/`.dll` if already built and
#' up to date; otherwise compiles `bifactor_kernel.c` with `R CMD SHLIB`
#' and loads the result. Safe to call repeatedly (no-op if already loaded).
#'
#' The clusters processed by `bifactor_negloglik()` are statistically
#' independent, so the kernel parallelizes the cluster loop with OpenMP
#' when available. This function first tries to compile with
#' `PKG_CFLAGS=-fopenmp PKG_LIBS=-fopenmp`; if that fails (e.g. plain Xcode
#' clang on macOS with no `libomp` installed, or a compiler that otherwise
#' doesn't support `-fopenmp`), it automatically falls back to a plain
#' (single-threaded, but still fully correct) build rather than erroring
#' out. Set `options(bifactor.use_openmp = FALSE)` before calling this to
#' skip the OpenMP attempt entirely and go straight to the plain build.
#' @keywords internal
.load_bifactor_kernel <- function(kernel_path) {
  if (is.loaded("bifactor_negloglik")) return(invisible(TRUE))

  if (!file.exists(kernel_path)) stop("kernel_path not found: ", kernel_path)

  if (grepl("\\.c$", kernel_path)) {
    dir  <- normalizePath(dirname(kernel_path), mustWork = TRUE)
    file <- basename(kernel_path)
    so_name <- paste0(tools::file_path_sans_ext(file), .Platform$dynlib.ext)
    so_path <- file.path(dir, so_name)

    need_compile <- !file.exists(so_path) ||
      file.mtime(file.path(dir, file)) > file.mtime(so_path)

    if (need_compile) {
      wd <- getwd()
      setwd(dir)
      on.exit(setwd(wd), add = TRUE)

      # Compile with the given PKG_CFLAGS/PKG_LIBS (R CMD SHLIB reads these
      # env vars even without a src/Makevars file), restoring whatever was
      # there before once we're done.
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
        message("Note: compiling bifactor_kernel.c with OpenMP (-fopenmp) ",
                "failed; retrying a single-threaded build (correct, just ",
                "not parallelized across clusters). Original compiler ",
                "output:\n", paste(res, collapse = "\n"))
        res <- build_with("-O2", "")
      }
      if (!file.exists(so_name)) {
        stop("R CMD SHLIB failed:\n", paste(res, collapse = "\n"))
      }
    }
    dyn.load(so_path)
  } else {
    dyn.load(kernel_path)
  }
  invisible(TRUE)
}
