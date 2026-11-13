#' @title Margins  cdf/pdf
#' @description This function computes the cdf and pdf
#' @param z     vector of responses
#' @param th    2-d link (can be negative)
#' @param model model for margin: "binomial" (bernoulli), "poisson", "nbinom" (mean is the parameter),"nbinom1" (p is the parameter), "geometric", "multinomial", "exponential", "weibull", "normal","t", "laplace"
#' @param dfM   degrees of freedom for the Student margin (default is 0)
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#' @examples
#' out = margins0(c(1:10),c(2.5,0),"poisson")
#' @export

margins0 = function(z,th,model,dfM){
  switch(model,
         "bernoulli" =,
         "binomial" = {

           out= berncpdf(z,th[1])
         },

         "poisson" = {
           out = poiscpdf(z,th[1])
         },

         "nbinom" = {
           out = nbinomcpdf(z,th)

         },

         "nbinom1" = {
           out = nbinom1cpdf(z,th)

         },

         "geometric" = {
           out = geomcpdf(z,th[1])

         },

         "exponential" = {
           out = expcpdf(z,th[1])
         },

         "weibull" = {
           out = weibcpdf(z,th)
         },
         "gaussian" =,
         "normal" = {
           out = normcpdf(z,th)
         },

         "t" = {
           out = tcpdf(z,th,dfM)
         },

         "laplace" = {
           out = lapcpdf(z,th)
         }

  )
  return(out)

}


#' @title Bernoulli with p = 1/(1+exp(-th)) cdf/pdf and derivatives
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th linear combination of covariates (can be negative)
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf and pdf with derivative with respect to parameters}
#'
t
#'
berncpdf=function(z,th){
  n=length(z)
  p = 1/(1+exp(-th))
  ind1 = (z==1)
  ind0 = (z==0)
  pdf  = vector(mode='numeric',n)
  cdf  = vector(mode='numeric',n)
  pdf[ind1]  = p
  pdf[ind0]  = 1-p
  cdf[ind1]  = 1
  cdf[ind0]  = 1-p


  out=cbind(cdf,pdf)
  return(out)
}


#' @title Poisson cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th values of lambda >0
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
poiscpdf=function(z,th){
  cdf=ppois(z,th)
  pdf= dpois(z,th)
  out=cbind(cdf,pdf)
  return(out)
}

#' @title Negative binomial cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is size > 0 and th[,2] is p, with 0<p<1; size  does not have to be integer
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
nbinomcpdf=function(z,th){
  n = length(z)
  if(is.vector(th)) th  = matrix(th, ncol=2, nrow=n, byrow=TRUE)
  size=th[,1]
  p = th[,2]

  cdf=pnbinom(z,size=size,prob=p)
  pdf=dnbinom(z,size=size,prob=p)


  out=cbind(cdf,pdf)
  return(out)
}
#' @title Negative binomial cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is size > 0 and th[,2] is mean > 0; size  does not have to be integer
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }


nbinom1cpdf=function(z,th){
  n = length(z)
  if(is.vector(th)) th  = matrix(th, ncol=2, nrow=n, byrow=TRUE)
  size=th[,1]
  mu = th[,2]

  p = size/(size+mu)

  cdf=pnbinom(z,size=size,mu=mu)
  pdf=dnbinom(z,size=size,mu=mu)


  out=cbind(cdf,pdf)
  return(out)
}

#' @title Geometric with p = 1/(1+exp(-th)) cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th linear combination of covariates (can be negative)
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
geomcpdf=function(z,th){

  p = 1/(1+exp(th))

  cdf  = pgeom(z,p)
  pdf  = dgeom(z,p)

  out=cbind(cdf,pdf)
  return(out)
}



#' @title Exponential cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th is mean > 0
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
expcpdf=function(z,th){
  if(is.vector(th)){ th = matrix(th,ncol=1)}
  th1=th[,1]
  ex=exp(-z*th1)
  cdf=1-ex
  pdf=th1*ex
  out=cbind(cdf,pdf)
  return(out)
}

#' @title Weibul cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is rate>0, th[,2] is shape  > 0;
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
weibcpdf=function(z,th){
  if(is.vector(th)){ la=th[1]; al=th[2] }
  if(is.matrix(th)){ la=th[,1]; al=th[,2] }
  #la=th[1]
  #al=th[2]
  za1= (z*la)^(al-1)
  za = za1*z*la
  lx = log(la*z)
  ex=exp(-za)
  cdf=1-ex
  pdf=al*la*za1*ex
  out= cbind(cdf,pdf)
 return(out)
}

#' @title normal cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is mean, th[,2] is standard deviation  > 0;
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
normcpdf=function(z,th){
  if(is.vector(th)){ m=th[1]; s=th[2] }
  if(is.matrix(th)){ m=th[,1]; s=th[,2] }
  stz = (z-m)/s
  stz2= stz*stz
  cdf=pnorm(z,mean=m,sd=s)
  pdf=exp(-.5*stz2)/sqrt(2*pi)/s
  out=cbind(cdf,pdf)
  return(out)
}

#' @title Student  cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is mean, th[,2] is standard deviation  > 0
#' @param df degrees of freedom
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
tcpdf=function(z,th,df){
  if(is.vector(th)){ m=th[1]; s=th[2] }
  if(is.matrix(th)){ m=th[,1]; s=th[,2] }
  df=df[1]
  stz = (z-m)/s
  stz2= stz*stz/df
  cdf=pt(stz,df=df)
  C=gamma((df+1)/2)/sqrt(df*pi)/gamma(df/2)
  pdf=C*(1+stz2)^(-(df+1)/2)/s
  out=cbind(cdf,pdf)
  return(out)
}


#' @title Laplace cdf/pdf and ders
#' @description This function computes the cdf, pdf, and associated derivatives
#' @param z vector of responses
#' @param th th[,1] is mean, th[,2] is standard deviation  > 0
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{out}{Matrix of conditional cdf, derivative with respect to parameter, pdf, }
#'
lapcpdf=function(z,th){
  if(is.vector(th)){ m=th[1]; s=th[2] }
  if(is.matrix(th)){ m=th[,1]; s=th[,2] }
  stz = (z-m)/s

  cdf=plap(stz)
  pdf=dlap(stz)/s

  out=cbind(cdf,pdf)
  return(out)
}

#' @title Laplace quantile
#' @description This function computes the qunatiles of Laplace distribution
#' @param u vector of probabilities
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{q}{Laplace quantiles}
#'
qlap = function(u){
  ind0=(u<=0.5)
  ind1 = (u>0.5)
  n = length(u)
  q=rep(0,n)
  q[ind0] = log(2*u[ind0])
  q[ind1] = -log(2*(1-u[ind1]))
  return(q)
}

#' @title Laplace cdf
#' @description This function computes the cdf of Laplace distribution
#' @param x vector of responses
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{f}{Laplace cdf}
#'
plap = function(x){
  ind0=(x<=0)
  ind1 = (x>0)
  n = length(x)
  f=rep(0,n)
  f[ind0] = 0.5*exp(x[ind0])
  f[ind1] = 1-0.5*exp(-x[ind1])
  return(f)
}

#' @title Laplace pdf
#' @description This function computes the pdf of Laplace distribution
#' @param x vector of responses
#'
#' @author Pavel Krupskii and Bruno N. Remillard, January 20, 2022
#' @return \item{f}{Laplace cdf}
#'
dlap = function(x){
  return(0.5*exp(-abs(x)))
}
