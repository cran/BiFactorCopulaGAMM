#' @title Link to copula parameter
#'
#' @description Computes the copula parameters given a linear combination of covariates.
#'
#' @param cpar    copula parameters
#' @param family  copula family: "normal (gaussian)" , "t" , "clayton" ,  "frank" , "gumbel", "fgm", "galambos", "huesler-reiss", "joe", "plackett", "ind" .
#'
#' @return \item{th}{unrestricted copula parameters}
#'
#' @details NOTE (fix): the previous version grouped `"clayton"` together
#' with `"huesler-reiss"`/`"galambos"` under `th = log(cpar)`. That's the
#' correct inverse for huesler-reiss/galambos (whose forward link in
#' `linkCop()` is `cpar = 1/expf = exp(th)`), but NOT for clayton, whose
#' forward link is instead `cpar = 2*tau/(1-tau)` with
#' `tau = 1/(1+exp(-th))` (the SAME tau clayton shares with gumbel/joe/
#' plackett, just combined into a different final formula) -- inverting
#' that gives `th = log(cpar/2)`, not `log(cpar)`. The old formula was
#' systematically wrong by a factor of exactly 2 in `cpar` (equivalently
#' `log(2)` in `th`) for every clayton value: `linkCop(log(cpar),
#' "clayton")$cpar` evaluates to `2*cpar`, not `cpar`. Also added the
#' `"fgm"` case, previously missing entirely even though `linkCop()`
#' supports it (`cpar = tanh(th)`, the same link as gaussian/t).
#'
#' @examples
#' th = cpar2link(0.5,"gaussian")
#' @export

cpar2link = function(cpar,family="clayton")
{


if(family=="normal"){family="gaussian"}

switch(family,
       "t"=,
       "normal"=,
       "gaussian" =,
       "fgm" = { th = atanh(cpar)},

       "frank" = {th=cpar },

       "huesler-reiss"=,
       "galambos" = { th = log(cpar) },

       "clayton" = { th = log(cpar/2) },  # NOTE (fix): see @details -- was wrongly grouped with huesler-reiss/galambos above

       "joe" =,
       "plackett"=,
       "gumbel" = { th = log(cpar-1) },
       "ind"    = {th = 0}

          )


return(th)
}
