#' @title Quantiles of margins
#' @description This function computes the quantiles of a given margin family
#' @param U     vector of probabilities
#' @param param parameters
#' @param family model for margin: "binomial" (bernoulli), "poisson", "nbinom" (mean is the parameter),"nbinom1" (p is the parameter), "geometric", "multinomial", "exponential", "weibull", "normal","t", "laplace"
#' @param size standard deviation if parameter of the distribution
#' @param shape only for Weibull distribution
#' @param dfM   degrees of freedom for the Student margin (default is NULL)
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
#' @examples
#' out = qmargin(0.5,"normal", param=c(10,2))
#'
#' @export

qmargin=function(U,family,param,size=NULL,shape=NULL,dfM=NULL){

  # NOTE (fix): the nbinom/nbinom1/geometric/exponential/weibull branches
  # below previously assigned their result to `out`, but this function
  # returns `y` -- so calling qmargin() for any of those five families
  # raised "object 'y' not found" (or silently returned a stale `y` left
  # over from an earlier call in the same environment/session). Fixed to
  # assign `y` consistently, matching every other branch.

  switch(family,
         "bernoulli" =,
         "binomial" = {
           p = param[1] #P(Y=1)
           y= as.numeric(U>1-p)
         },

         "poisson" = {
           mu  = param[1]
           y = qpois(U,mu)
         },

         "nbinom" = {  #mean is the parameter
           mu  = param[1]
           y = qnbinom(U,size=param[2],mu=mu)
         },

         "nbinom1" = { # p is the parameter
           p = param[1]
           y = qnbinom(U,size=param[2],prob=p)
         },

         "geometric" = {
           p = param[1]
           y = qgeom(U,p)
         },


         "exponential" = {
           mean = param[1]
           y = -mean*log(1-U)
         },

         "weibull" = {
           shape  = param[1]
           y = qweibull(U,shape,scale=param[2])
         },
         "normal" =,
         "gaussian" = {
           y = param[1]+param[2]*qnorm(U)
         },

         "t" = {
           y = param[1]+param[2]*qt(U,dfM)
         },

         "laplace" = {
           y = param[1]+param[2]*qlap(U)
         }


  )
return(y)
}
