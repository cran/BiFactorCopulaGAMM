#' @title Two-step ("IFM"-style) estimation for the bi-factor copula GAMM
#'
#' @description Implements the two-step estimator suggested by section 2.1
#' of the bi-factor copula paper: "the parameters of Gj and Dj,1 can be
#' estimated using the CRAN package CopulaGAMM ... with no covariates,"
#' applied separately to every margin (Step 1, via `EstMargin1Factor()`,
#' with an analytic gradient built from `margins()`/`coplik()`), followed
#' by plugging those fixed values into the full bi-factor likelihood to
#' estimate only the copula-2 parameters (Step 2, reusing the existing C
#' kernel, `bifactor_negloglik()`, but now optimizing over just `dp`
#' parameters instead of the whole model).
#'
#' This changes the *shape* of the optimization, not just its constant:
#' Step 1 replaces one big, badly-scaled joint optimization with `dp`
#' small, cheap, single-quadrature ones (no integral over the second
#' latent factor W at all); Step 2 keeps the double quadrature but only
#' has to differentiate `dp` parameters instead of the full parameter
#' count. Both are typically much cheaper than one joint fit over
#' everything at once.
#'
#' IMPORTANT CAVEAT: two-step (IFM-style) estimators are consistent but
#' not fully efficient -- standard errors read off Step 2 alone understate
#' the true uncertainty, because they treat the Step 1 estimates as known
#' constants rather than propagating their sampling variability. If you
#' need valid inference (not just point estimates), either use a
#' sandwich/Godambe covariance, bootstrap over clusters, or (recommended,
#' and the default here) use these estimates purely as a warm start for
#' one final joint optimization over all parameters (`refine=TRUE`):
#' since it starts essentially at the optimum, that final pass typically
#' converges in a handful of iterations rather than a full cold-start
#' search, so you get IFM-level speed for the hard part and exact joint-
#' MLE point estimates/standard errors at the end.
#'
#' @inheritParams EstBifactorCopulaGAMMFast
#' @param Y     matrix of responses (assumed continuous or discrete).
#' @param X     matrix of covariates (assumed continuous or discrete).
#' @param clu   variable of size n defining the clusters; can be a factor
#' @param model list of lists for columns vectors of x. For any j, model[[j]]$margin is a list containing the margin family, param (vector of parameters), and size if necessary.
#' @param start list of vector of starting point for the estimation.
#' @param LB    list of vector of lower bound for the parameters.
#' @param UB    list of vector of upper bound for the parameters.
#' @param nq    number of nodes and weights for Gaussian quadrature of the product of conditional copulas; default is 25.
#' @param prediction logical variable for prediction of latent variables V (default is TRUE).
#' @param refine logical (default TRUE). If TRUE, runs one final joint
#'   `EstBifactorCopulaGAMMFast()` optimization over *all* parameters,
#'   warm-started from the Step 1 + Step 2 estimates. If FALSE, returns the
#'   two-step estimates directly (faster, but see the efficiency caveat
#'   above).
#' @param compute_se logical (default TRUE). If TRUE, also computes
#'   `se`/`tval`/`pval` (passed through to `EstMargin1Factor()` and, if
#'   `refine=TRUE`, to `EstBifactorCopulaGAMMFast()` too, plus this
#'   function's own Hessian for cpar2 when `refine=FALSE`) -- several
#'   extra O(n^2)-ish Hessian evaluations on top of the estimation
#'   itself. Set to FALSE to skip all of that and get only point
#'   estimates back faster (`se`/`tval`/`pval` will be `NULL`).
#'
#' @return list with `coef` (list with `copulas`/`margins` matrices, same
#'   shape as `EstBifactorCopulaGAMMFast()`'s `coef` -- `margins` is in
#'   natural units via `link2parM()`, matching that function's own
#'   convention, not the raw pre-link scale Step 1 optimizes over) and
#'   `gradient` (shaped like `coef`: a list with `copula` and `margin`
#'   matrices, each `dp x 2` with the same row order as `coef$copulas`/
#'   `coef$margins`, rather than a flat vector. Same parameter scale as
#'   `EstBifactorCopulaGAMMFast()`'s `gradient` -- Step 1's copula-1
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
EstBifactorTwoStep <- function(Y, X = NULL, clu, model, start = NULL,
                                LB = NULL, UB = NULL, nq = 25,
                                prediction = TRUE,
                                kernel_path = "bifactor_kernel.c",
                                refine = TRUE, compute_se = TRUE) {

  t0 <- Sys.time()

  ## ---- setup: identical bookkeeping to EstBifactorCopulaGAMMFast() ----
  dismod  <- c("binomial", "bernoulli", "poisson", "nbinom", "geometric", "multinomial")
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
  nn <- vapply(seq_len(K), function(k) sum(clu == clusters[k]), integer(1))

  cont  <- rep(1, dp)
  rot   <- matrix(0, ncol = 2, nrow = dp)
  margin <- NULL
  cop   <- NULL
  dfM   <- rep(0, dp)
  dfC   <- matrix(0, ncol = 2, nrow = dp)

  for (j in 1:d) {
    mar <- model$Y[[j]]$margin$family
    margin <- c(margin, mar)
    df <- model$Y[[j]]$margin$dfM
    if (!is.null(df)) dfM[j] <- df
    if (mar %in% dismod) cont[j] <- 0
    mod <- model$Y[[j]]
    cop <- c(cop, mod$copula[[1]]$family, mod$copula[[2]]$family)
    df1 <- mod$copula[[1]]$dfC; if (!is.null(df1)) dfC[j, 1] <- df1
    df2 <- mod$copula[[2]]$dfC; if (!is.null(df2)) dfC[j, 2] <- df2
    rot1 <- mod$copula[[1]]$rot; rot[j, 1] <- if (is.null(rot1)) 0 else rot1
    rot2 <- mod$copula[[2]]$rot; rot[j, 2] <- if (is.null(rot2)) 0 else rot2
  }
  if (dp == 1) cop[2] <- "ind"
  if (dp > d) {
    for (j in 1:p) {
      jj <- j + d
      margin <- c(margin, model$X[[j]]$margin$family)
      df <- model$X[[j]]$margin$dfM
      if (!is.null(df)) dfM[jj] <- df
      if (margin[jj] %in% dismod) cont[jj] <- 0
      mod <- model$X[[j]]
      cop <- c(cop, mod$copula[[1]]$family, mod$copula[[2]]$family)
      df1 <- mod$copula[[1]]$dfC; if (!is.null(df1)) dfC[jj, 1] <- df1
      df2 <- mod$copula[[2]]$dfC; if (!is.null(df2)) dfC[jj, 2] <- df2
      rot1 <- mod$copula[[1]]$rot; rot[jj, 1] <- if (is.null(rot1)) 0 else rot1
      rot2 <- mod$copula[[2]]$rot; rot[jj, 2] <- if (is.null(rot2)) 0 else rot2
    }
  }

  dd <- as.integer(vapply(seq_len(dp), function(j) param.dim(margin[j]), numeric(1)))
  XX <- as.matrix(cbind(Y, X))

  if (is.null(start) || is.null(LB) || is.null(UB)) {
    outm <- bounds.mar(XX, margin)
    outc <- bounds.cop(cop)
    start <- c(outm$start, outc$start)
    UB <- c(outm$UB, outc$UB)
    LB <- c(outm$LB, outc$LB)
  }

  dm <- sum(dd)
  dm_off <- c(0, cumsum(dd))
  margin_idx <- function(j) (dm_off[j] + 1):(dm_off[j] + dd[j])
  cpar1_idx  <- function(j) dm + 2 * j - 1
  cpar2_idx  <- function(j) dm + 2 * j

  ## ---- Step 1: per-margin, single-quadrature, analytic-gradient fits ----
  t1 <- Sys.time()
  step1 <- vector("list", dp)
  parM  <- matrix(0, ncol = 2, nrow = dp)
  cpar1_fixed <- rep(0, dp)

  for (j in seq_len(dp)) {
    m_idx <- margin_idx(j)
    c1_idx <- cpar1_idx(j)

    # Convert the joint model's pre-link start/LB/UB for the copula-1 slot
    # into raw copula-parameter space (the scale EstMargin1Factor works in)
    # by pushing them through linkCop() -- reusing bounds.cop()'s existing,
    # already-sane bounds instead of guessing per-family raw ranges here.
    raw <- vapply(c(start[c1_idx], LB[c1_idx], UB[c1_idx]),
                   function(v) linkCop(v, cop[2 * j - 1])$cpar, numeric(1))
    raw_start <- raw[1]
    raw_LB <- min(raw[2], raw[3])
    raw_UB <- max(raw[2], raw[3])

    start1 <- c(start[m_idx], raw_start)
    LB1    <- c(LB[m_idx], raw_LB)
    UB1    <- c(UB[m_idx], raw_UB)

    fit_j <- EstMargin1Factor(
      z = XX[, j], clu = clu, margin = margin[j], cop1 = cop[2 * j - 1],
      rot1 = rot[j, 1], dfM = dfM[j], dfC1 = dfC[j, 1],
      nq = nq, start = start1, LB = LB1, UB = UB1, compute_se = compute_se
    )
    step1[[j]] <- fit_j

    nth <- dd[j]
    parM[j, 1:nth] <- fit_j$param[1:nth]
    cpar1_fixed[j] <- fit_j$param[nth + 1]
  }
  step1_time <- Sys.time() - t1

  ## ---- fixed F1/F2/f from the Step-1 margin estimates ----
  F1 <- matrix(0, ncol = dp, nrow = N)
  F2 <- F1
  f  <- matrix(1, ncol = dp, nrow = N)
  for (j in seq_len(dp)) {
    out0 <- margins0(XX[, j], parM[j, ], margin[j], dfM = dfM[j])
    F1[, j] <- out0[, 1]
    f[, j]  <- out0[, 2]
    if (cont[j] == 0) { F2[, j] <- F1[, j] - f[, j]; f[, j] <- 1 }
  }

  ## ---- integer codes / cluster grouping for the C kernel (same as Fast) ----
  ord <- unlist(lapply(seq_len(K), function(k) which(clu == clusters[k])))
  order0 <- as.integer(ord - 1L)
  cbound <- as.integer(c(0, cumsum(nn)))

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
  rot1v <- as.integer(rot[, 1]); rot2v <- as.integer(rot[, 2])
  dfC1v <- as.double(dfC[, 1]); dfC2v <- as.double(dfC[, 2])
  contv <- as.integer(cont)
  gl <- statmod::gauss.quad.prob(nq)
  nl_d <- as.double(gl$nodes); wl_d <- as.double(gl$weights)
  N_i <- as.integer(N); dp_i <- as.integer(dp); K_i <- as.integer(K); nq_i <- as.integer(nq)

  .load_bifactor_kernel(kernel_path)

  ## ---- Step 2: optimize ONLY cpar2 (dp parameters), cpar1/F1/F2/f fixed ----
  t2 <- Sys.time()
  start2 <- vapply(seq_len(dp), function(j) start[cpar2_idx(j)], numeric(1))
  LB2    <- vapply(seq_len(dp), function(j) LB[cpar2_idx(j)], numeric(1))
  UB2    <- vapply(seq_len(dp), function(j) UB[cpar2_idx(j)], numeric(1))

  likf2 <- function(param2) {
    if (any(param2 < LB2) || any(param2 > UB2)) return(1e100)
    cpar2 <- vapply(seq_len(dp), function(j) linkCop(param2[j], cop[2 * j])$cpar, numeric(1))
    res <- .C("bifactor_negloglik",
              F1 = as.double(F1), F2 = as.double(F2), f = as.double(f),
              N = N_i, dp = dp_i, cont = contv,
              fam1 = fam1, fam2 = fam2, rot1 = rot1v, rot2 = rot2v,
              cpar1 = as.double(cpar1_fixed), cpar2 = as.double(cpar2),
              dfC1 = dfC1v, dfC2 = dfC2v,
              order0 = order0, cbound = cbound, K = K_i,
              nl = nl_d, wl = wl_d, nq = nq_i, out = double(1))
    # Defensive guard: see EstBifactorCopulaGAMMFast.R's likf() for why.
    if (!is.finite(res$out)) return(1e100)
    res$out
  }
  grad2 <- function(param2) .central_grad(likf2, param2, LB2, UB2)
  fit2 <- nlminb(start = start2, objective = likf2, gradient = grad2,
                  lower = LB2, upper = UB2,
                  control = list(iter.max = 200, eval.max = 400))
  # nlminb()'s return object has no `$gradient` field (only par,
  # objective, convergence, iterations, evaluations, message) -- unlike
  # the custom `mle` list built in EstBifactorCopulaGAMM(Fast)(), which
  # explicitly re-evaluates the gradient function at the solution. Do
  # the same here, since fit2$gradient is read below (and by the caller).
  fit2$gradient <- grad2(fit2$par)
  step2_time <- Sys.time() - t2

  ## ---- assemble a full-model parameter vector, coef, and gradient ----
  # Always done now (previously only under refine=TRUE, deferred because
  # of the invfunc() bug that's since been fixed -- this is cheap
  # (bisection on `dp` scalars), so there's no reason to skip it).
  #
  # coef: same shape as EstBifactorCopulaGAMMFast()'s `coef` (copulas,
  # margins matrices).
  #
  # gradient: same PARAMETER SCALE as EstBifactorCopulaGAMMFast()'s
  # `gradient` (i.e. w.r.t. the pre-link "param" values, not the raw
  # copula parameters) -- not just Step 1's and Step 2's gradients
  # concatenated as-is. Step 2's cpar2 gradient (`fit2$gradient`) is
  # already on that scale (nlminb optimizes param2 directly). Step 1's
  # cpar1 gradient is NOT -- EstMargin1Factor() optimizes the raw
  # copula-1 parameter directly, so its gradient is d(negloglik)/d(cpar1),
  # not d(negloglik)/d(param). Converted via the chain rule,
  # d(negloglik)/d(param) = d(negloglik)/d(cpar1) * d(cpar1)/d(param),
  # using `.cop_link_deriv()` (a numeric derivative of linkCop()$cpar,
  # defined in EstBifactorCopulaGAMMFast.R) for the second factor --
  # more robust than relying on linkCop()'s own `hder` field, whose
  # exact convention isn't independently verified here.
  warm_margin <- unlist(lapply(seq_len(dp), function(j) parM[j, 1:dd[j]]))
  warm_cop  <- numeric(2 * dp)
  grad_cop  <- numeric(2 * dp)
  parC_2step <- matrix(0, ncol = 2, nrow = dp)
  for (j in seq_len(dp)) {
    fam1j <- cop[2 * j - 1]
    fn <- function(v, th) vapply(seq_along(v), function(i) linkCop(v[i], th[i])$cpar, numeric(1))
    param_cpar1 <- invfuncNew(cpar1_fixed[j], fn, fam1j,
                            lb = LB[cpar1_idx(j)], ub = UB[cpar1_idx(j)])
    warm_cop[2 * j - 1] <- param_cpar1
    warm_cop[2 * j]     <- fit2$par[j]

    dcpar1_dparam <- .cop_link_deriv(param_cpar1, fam1j)
    grad_cop[2 * j - 1] <- step1[[j]]$gradient$cpar1 * dcpar1_dparam
    grad_cop[2 * j]     <- fit2$gradient[j]

    parC_2step[j, 1] <- cpar1_fixed[j]
    parC_2step[j, 2] <- linkCop(fit2$par[j], cop[2 * j])$cpar
  }
  warm_start <- c(warm_margin, warm_cop)
  warm_start <- pmin(pmax(warm_start, LB), UB)  # guard against bisection edge misses

  grad_margin <- unlist(lapply(seq_len(dp), function(j) step1[[j]]$gradient$margin))
  # `parM` holds the RAW (pre-link) theta values from Step 1 -- correct
  # for margins0() (line ~188) and for warm_margin/warm_start above,
  # both of which need that same unconstrained scale. But EstBifactor-
  # CopulaGAMMFast()'s own `coef$margins` reports the natural-units
  # estimate via link2parM() (e.g. Poisson: exp(theta) -> the actual
  # rate; Bernoulli: the logit -> the actual probability) -- match that
  # here rather than leaking the raw optimizer scale into `coef`.
  parM_natural <- matrix(0, ncol = 2, nrow = dp)
  for (j in seq_len(dp)) parM_natural[j, ] <- link2parM(parM[j, ], margin[j])
  coef_2step <- list(copulas = parC_2step, margins = parM_natural)
  # Per-copula gradient, Step 1 (cpar1) then Step 2 (cpar2), one row per
  # margin/covariate j -- same dp x 2 shape/row-order as coef$copulas,
  # just reshaping grad_cop (already laid out as (cpar1_1,cpar2_1,
  # cpar1_2,cpar2_2,...)) into a matrix instead of a flat vector.
  gradient_copulas_2step <- matrix(grad_cop, ncol = 2, byrow = TRUE,
                                    dimnames = list(NULL, c("step1", "step2")))
  # NOTE: -fit2$objective is already the FULL joint (double-quadrature,
  # all dp margins) negative log-likelihood evaluated at the two-step
  # estimate (cpar1/margin params fixed from Step 1, cpar2 from Step 2)
  # -- likf2() calls the same bifactor_negloglik() kernel the joint model
  # uses. Step 1's per-margin `loglik` values are each a DIFFERENT,
  # restricted (single-quadrature, one-margin) likelihood and must not be
  # added on top of this -- they're not additive components of it.
  loglik_2step <- -fit2$objective
  npar_total <- length(warm_start)
  aic_2step <- 2 * npar_total - 2 * loglik_2step
  bic_2step <- npar_total * log(N) - 2 * loglik_2step

  final <- NULL
  refine_time <- as.difftime(0, units = "secs")
  if (isTRUE(refine)) {
    t3 <- Sys.time()
    final <- EstBifactorCopulaGAMMFast(
      Y = Y, X = X, clu = clu, model = model,
      start = warm_start, LB = LB, UB = UB, nq = nq,
      prediction = prediction, kernel_path = kernel_path, method = "nlminb",
      compute_se = compute_se
    )
    refine_time <- Sys.time() - t3
  }

  # Top-level coef/gradient/loglik/aic/bic: prefer the refined joint fit
  # when available (it's the authoritative full-MLE result); otherwise
  # the two-step values assembled above. `gradient` is shaped like
  # `coef` -- a list with `copula` and `margin` matrices (dp x 2 each,
  # same row order as coef$copulas/coef$margins) -- rather than a flat
  # vector, in both cases. `.margin_grad_matrix()` reshapes a flat
  # margin-block gradient (length sum(dd), in the same per-margin order
  # as `margin_idx()`) into that dp x 2 form, NA-padded for margins with
  # only one real parameter (matching how coef$margins/parM are built).
  .margin_grad_matrix <- function(flat_margin_grad) {
    # NA (not 0) for a margin's unused second column: dd[j]==1 means
    # there IS no second parameter for that margin, so no gradient was
    # ever computed for it -- 0 would misleadingly read as "gradient is
    # zero" (a stationary point) rather than "not applicable here".
    m <- matrix(NA_real_, ncol = 2, nrow = dp)
    for (j in seq_len(dp)) m[j, 1:dd[j]] <- flat_margin_grad[margin_idx(j)]
    m
  }

  # ---- standard errors / t-values / p-values (two-step, naive; optional
  # via compute_se, since this needs its own Hessian for cpar2 on top of
  # whatever EstMargin1Factor()/EstBifactorCopulaGAMMFast() already did) ----
  if (compute_se) {
    # Margin and cpar1 standard errors are REUSED directly from
    # EstMargin1Factor()'s own se$margin/se$cpar1 (already natural units,
    # already delta-method-converted there) -- no need to redo that work.
    # cpar2 needs its own Hessian, since Step 2 (likf2/param2) is a
    # separate fit EstMargin1Factor() never sees.
    #
    # Defensive check: this relies on EstMargin1Factor() returning
    # `se$margin`/`se$cpar1` on every step1[[j]] result (i.e. that it was
    # also called with compute_se=TRUE, as it is above). If an older
    # copy of EstMargin1Factor.R (from before it computed se/tval/pval)
    # is still sourced, step1[[j]]$se is NULL, and step1[[j]]$se$cpar1 is
    # then also NULL (R allows `$` on NULL) -- a length-0 value, which
    # produces a cryptic "number of items to replace is not a multiple of
    # replacement length" a few lines below instead of saying what's
    # actually wrong. Catch it here with a clear, actionable message.
    missing_se <- vapply(step1, function(s) is.null(s$se) || is.null(s$se$cpar1), logical(1))
    if (any(missing_se)) {
      stop("step1[[", which(missing_se)[1], "]]$se is missing (or incomplete). ",
           "This means the EstMargin1Factor() actually sourced in this session ",
           "doesn't compute se/tval/pval -- almost certainly an older copy of ",
           "EstMargin1Factor.R than the one this EstBifactorTwoStep.R expects. ",
           "Please re-source the latest EstMargin1Factor.R and try again.")
    }

    .margin_list_to_matrix <- function(get_j) {
      m <- matrix(NA_real_, ncol = 2, nrow = dp)
      for (j in seq_len(dp)) { v <- get_j(j); m[j, seq_along(v)] <- v }
      m
    }
    se_margin_2step <- .margin_list_to_matrix(function(j) step1[[j]]$se$margin)

    H2 <- .numerical_hessian_from_grad(grad2, fit2$par, LB2, UB2)
    V2 <- tryCatch(solve(H2), error = function(e) matrix(NA_real_, nrow(H2), ncol(H2)))
    d2 <- diag(V2)
    d2[!is.na(d2) & d2 < 0] <- NA_real_
    se_raw_cpar2 <- sqrt(d2)

    se_copula_2step <- matrix(NA_real_, ncol = 2, nrow = dp, dimnames = list(NULL, c("cpar1", "cpar2")))
    for (j in seq_len(dp)) {
      se_copula_2step[j, 1] <- step1[[j]]$se$cpar1
      dc2 <- .cop_link_deriv(fit2$par[j], cop[2 * j])
      se_copula_2step[j, 2] <- se_raw_cpar2[j] * abs(dc2)
    }
    se_2step <- list(copula = se_copula_2step, margin = se_margin_2step)
    # tval: an asymptotic Wald statistic (coef/se), not a finite-sample
    # Student-t, so pval uses the normal approximation.
    tval_2step <- list(copula = coef_2step$copulas / se_2step$copula,
                        margin = coef_2step$margins / se_2step$margin)
    pval_2step <- list(copula = 2 * pnorm(-abs(tval_2step$copula)),
                        margin = 2 * pnorm(-abs(tval_2step$margin)))
  } else {
    se_2step <- NULL; tval_2step <- NULL; pval_2step <- NULL
  }

  if (!is.null(final)) {
    coef <- final$coef
    # EstBifactorCopulaGAMMFast() now returns `gradient`/`se`/`tval`/
    # `pval` in this same list(copula=,margin=) shape itself -- reuse
    # them directly rather than re-deriving anything. These are the
    # authoritative full-MLE standard errors/tests, valid without the
    # naive-SE caveat below. (If compute_se=FALSE, final$se etc. are
    # already NULL, since compute_se was passed through to that call.)
    gradient <- final$gradient
    se <- final$se; tval <- final$tval; pval <- final$pval
    loglik <- final$loglik; aic <- final$aic; bic <- final$bic
    se_note <- if (compute_se) paste(
      "From the refined joint fit (refine=TRUE): valid full-MLE",
      "standard errors/tests."
    ) else "compute_se=FALSE: se/tval/pval were not computed."
  } else {
    coef <- coef_2step
    gradient <- list(
      copula = matrix(grad_cop, ncol = 2, byrow = TRUE,
                       dimnames = list(NULL, c("cpar1", "cpar2"))),
      margin = .margin_grad_matrix(grad_margin)
    )
    se <- se_2step; tval <- tval_2step; pval <- pval_2step
    loglik <- loglik_2step; aic <- aic_2step; bic <- bic_2step
    se_note <- if (compute_se) paste(
      "NAIVE two-step standard errors (refine=FALSE): margin and cpar1",
      "SEs come from each EstMargin1Factor() fit treated in isolation,",
      "and cpar2's SE treats margin/cpar1 as fixed/known -- neither",
      "propagates the OTHER stage's sampling variability, so these",
      "understate the true uncertainty (a well-known property of",
      "two-step/IFM estimators). Use refine=TRUE for valid inference."
    ) else "compute_se=FALSE: se/tval/pval were not computed."
  }

  total_time <- Sys.time() - t0
  list(
    coef = coef,
    gradient = gradient,
    se = se,
    tval = tval,
    pval = pval,
    se_note = se_note,
    gradient_copulas = gradient_copulas_2step,
    loglik = loglik,
    aic = aic,
    bic = bic,
    step1 = step1,
    step2 = list(param = fit2$par, loglik = -fit2$objective, fit = fit2,
                 cpar1_fixed = cpar1_fixed, cpar2 = vapply(seq_len(dp),
                   function(j) linkCop(fit2$par[j], cop[2 * j])$cpar, numeric(1))),
    warm_start = warm_start,
    final = final,
    calc_time = c(step1 = step1_time, step2 = step2_time,
                   refine = refine_time, total = total_time)
  )
}

#' @title (internal) Numerical Hessian from a gradient function
#' @description Local copy of the helper from EstBifactorCopulaGAMMFast.R
#' (this file previously relied on that one being sourced first, which
#' broke -- "could not find function" -- whenever an older copy of that
#' file, from before these helpers were added, was still loaded). Same
#' implementation as there: forms the Hessian of a scalar function at
#' `x` via central differences of its gradient, symmetrized afterward.
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

#' @title (internal) Numeric derivative of linkCop()'s cpar map
#' @description Local copy of the helper from EstBifactorCopulaGAMMFast.R
#' -- see `.numerical_hessian_from_grad()` above for why this file now
#' carries its own copy rather than relying on that file's. Returns
#' d(cpar)/d(param), via central difference of `linkCop(param,
#' family)$cpar`.
#' @keywords internal
.cop_link_deriv <- function(param_raw, family, h = 1e-6) {
  (linkCop(param_raw + h, family)$cpar - linkCop(param_raw - h, family)$cpar) / (2 * h)
}
