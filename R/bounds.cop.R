#' @title Starting values, upper and lower bounds for parameter estimation of bi-factor copulas
#' @description The starting point is a value giving approximately a Kendall's tau of 0.5,
#' and the upper and lower bounds are values giving a tau of 0.99 (resp. 0.1 or -0.99) for the copula families
#' @param cop Vector of copula families: "gaussian" (normal), "t" , "clayton" ,  "frank" , "gumbel", "galambos", "joe", "huesler-reiss", "plackett".
#' @returns{out} List of starting points (start), upper bounds (UB) and lower bounds (LB)
#'
#' @examples
#' cop=c("gumbel","gumbel","normal","clayton","frank","clayton")
#' out=bounds.cop
#'
#' @export
#'
bounds.cop <- function(cop){

  p = length(cop)
  start=rep(0,p)
  UB=rep(0,p)
  LB=UB
  for(j in 1:p){
    fam = cop[j]
    switch (fam,
      "gaussian" =,
      "normal" =,
      "t" ={
      #start[j]=0.5
      start[j]=0
      UB[j] = atanh(.99)  #  2.65
      LB[j] = -UB[j]      # -2.65
      },
      "frank"={
      start[j] = 0.5
      UB[j] =  400
      LB[j] = -400
      },
      "clayton" =,
      "gumbel" =
      {
      start[j]=0
      UB[j]=qlogis(.99) #  4.5
      LB[j]=-UB[j] # -4.5
      },
      "galambos" =
        {
        start[j]=log(1.285) #tau = 0.5
        UB[j] = log(44.43) #tau = 0.99
        LB[j]=log(0.16)    #tau = 0.01
        },
      "joe" =
        {
        start[j]=log(2.8563-1) #tau = 0.5
        UB[j] = log(60) #tau = 0.99
        LB[j]=log(0.4)    #tau = 0.01
        },
      "huesler-reiss" =
        {
        start[j]=log(1.28) #tau = 0.5
        UB[j] = log(44.43-1) #tau = 0.99
        LB[j]=log(1.0175-1)    #tau = 0.01
        },
      "plackett" =
        {
          start[j]=log(11.4-1) #tau = 0.5
          UB[j] = log(20000-1) #tau = 0.99
          LB[j]=log(1.045-1)    #tau = 0.01
        }
    )
  }
  out=list(start=start,LB=LB,UB=UB)
  out
}


