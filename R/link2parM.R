#' @title Parameter values from link parameter for margins
#' @description Computation of parameter values for margins from the associated link parameters
#' @param th  Link vector
#' @param marg Margin family: "binomial" (bernoulli), "poisson", "exponential",  "normal".
#'
#' @returns{parM} Associated margin parameters
#'
#' @examples
#' p=link2parM(.43,"bernoulli")
#'
#' @export
#'
link2parM <- function(th,marg){

    switch (marg,
            "gaussian" =,
            "normal" = {
              parM=c(th[1],th[2])
            },
            "poisson" =,
            "exponential"={
              parM =c(exp(th[1]),NA)

                           },
            "bernoulli" =,
            "binomial"  =
              {
                p = 1/(1+exp(-th[1]))
                parM = c(p,NA)
              }

    )

 return(parM)
}

