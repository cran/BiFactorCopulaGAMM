#' @title Starting values, upper and lower bounds for parameter estimation of margins
#' @description The starting point is an estimation of the parameters and the upper and lower bounds are approximate 99.7% CI for the unrestricted parameters
#' @param X  Matrix of data
#' @param margin Vector of margin families: "binomial" (bernoulli), "poisson",  "exponential", "normal"..
#' @returns{out} List of starting points (start), upper bounds (UB) and lower bounds (LB)
#'
#' @examples
#'  margin=c("bernoulli","poisson","normal")
#'  U=matrix(runif(300),ncol=3)
#'  X = cbind(as.double(U[,1]>.3),qpois(U[,2],5),10+2*qnorm(U[,3]))
#'  out=bounds.mar(X,margin)
#'
#' @export
#'
bounds.mar <- function(X,margin){

  m = colMeans(X)
  dd=dim(X)
  n = dd[1]
  p = dd[2]


  start=NULL
  UB=NULL
  LB=UB
  sn=3/sqrt(n)

  for(j in 1:p){
    fam = margin[j]


    switch (fam,
            "gaussian" =,
            "normal" = {
              sx = sd(X[,j])
              s=c(m[j],sx)
              E = sn*c(sx,sx)
            },
            "exponential"={
              s =-log(m[j])
              E = sn
              },
            "poisson" =
            {
              s = log(m[j])
              E = sn/sqrt(m[j])
            },
            "bernoulli" =,
            "binomial" =
              {
                s=qlogis(m[j])
                E = sn/sqrt(m[j]*(1-m[j]))
              }

    )
    start=c(start,s)
    UB=c(UB,s+E)
    LB=c(LB,s-E)
  }

  out=list(start=start,LB=LB,UB=UB)
  out
}

