#' @title Number of parameters
#' @description Gives the number of paramters for a given idstribution
#' @param family model for margin: "binomial" (bernoulli), "poisson", "nbinom" (mean is the parameter),"nbinom1" (p is the parameter), "geometric", "multinomial", "exponential", "weibull", "normal","t", "laplace"
#' @author Pavel Krupskii and Bruno N. Remillard, March 9, 2026
#'
#' @return \item{dd}{Number of parameters}
#'
#' @examples
#' dd = param.dim("normal")
#'
#' @export

param.dim=function(family){

  dd=2
  par1 = c("bernoulli","binomial","exponential","geometric","poisson")
  if(family %in% par1){dd=1}
return(dd)
}
