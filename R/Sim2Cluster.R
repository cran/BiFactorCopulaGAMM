#' @title Simulation of clustered data using bi-factor copulas
#'
#' @description Generate a random sample of observations from a bi-factor copula  mixed regression model.
#'
#' @param model list of lists for vector Y and vector X; model$Y is a list of lists of models for the copulas
#' and the margins and model$X is a list of list of models for the covariates; model$Y[[j]]$margin or model$X[[j]]$margin
#' is a list containing the margin family (as provided in copulaGAMM), param (vector of parameters), and size if necessary,
#' while model$Y[[j]]$copula[[1]] and model$Y[[j]]$copula[[2]] are lists defining the copulas (included in `copulaGAMM` package),
#' including the copula family name (family), the parameters (cpar), and the rotations (rot)
#' with values in 0 (default),90,180, or 270.
#' @param nclu    vector of sample sized for clusters
#'
#' @return \item{data}{Data frame with simulated values of Y,X V, and values of clusters}
#'
#' @author  Bruno N. Remillard
#'
#' @examples
#' nclu=rep(5,25) # 25 clusters of size 5
#' ccY=list(list(family="gumbel",param=2),list(family="gumbel",param=1.4,rot=90) )
#' modY1 = list(margin=list(family="bernoulli", param=c(0.6)), copula=ccY)
#' modY=list(modY1)
#' ccX1 = list(list(family="clayton",param=0.86,rot=180),list(family="clayton",param=0.2,rot=180))
#' ccX2 = list(list(family="gaussian",param=0.6),list(family="clayton",param=2))
#' modX1 = list(margin=list(family="bernoulli", param=c(0.6)),  copula=ccX1)
#' modX2 = list(margin=list(family="gaussian" , param=c(10,0.2)), copula=ccX2)
#' modX=list(modX1,modX2)
#' model=list(Y=modY,X=modX)
#' set.seed(123)
#' out=Sim2Cluster(model,nclu)
#'
#' @export
#'
Sim2Cluster<-function(model,nclu)
{
  N = sum(nclu)
  K = length(nclu) # number of clusters
  modY = model$Y
  d = length(modY)
  modX = model$X
  p = length(modX)
  U=NULL
  Utilde=NULL
  V=NULL
  clu=NULL
  Y=matrix(0,ncol=d,nrow=N)
  X=matrix(0,ncol=p,nrow=N)
  for(k in 1:K){
    nn=nclu[k]
    W = stats::runif(nn)
    u1=stats::runif(d*nn)
    u2=stats::runif(p*nn)

    Wtilde=matrix(u1,ncol=d)
    Wcheck=matrix(u2,ncol=p)
    Ztilde=NULL
    Z=NULL
    Z1=rep(0,nn)
    Z2=rep(0,nn)
    V0=stats::runif(1)
    for(j in 1:d)
    {
      mod = model$Y[[j]]
      rot1 = mod$copula[[1]]$rot
      if(is.null(rot1)){rot1=0}
      rot2 = mod$copula[[2]]$rot
      if(is.null(rot2)){rot2=0}
      for(i in 1:nn)
      {
      # NOTE: calls qcondNew() (this package's own fixed dispatcher), not
      # CopulaGAMM::qcond() -- see qcondNew.R's own NOTE for why the two
      # aren't interchangeable.
      Z2[i] = qcondNew(Wtilde[i,j],W[i], family = mod$copula[[2]]$family,cpar=mod$copula[[2]]$param,rot=rot2)

      Z1[i] = qcondNew(Z2[i],V0,family= mod$copula[[1]]$family,cpar=mod$copula[[1]]$param,rot=rot1)
      }
      Ztilde=c(Ztilde,Z1)
    }
    Ztilde = matrix(Ztilde,ncol=d,byrow=FALSE)
    for(j in 1:p)
    {
      mod = model$X[[j]]
      rot1 = mod$copula[[1]]$rot
      if(is.null(rot1)){rot1=0}
      rot2 = mod$copula[[2]]$rot
      if(is.null(rot2)){rot2=0}
      for(i in 1:nn){
      Z2[i] = qcondNew(Wcheck[i,j],W[i], family = mod$copula[[2]]$family,cpar=mod$copula[[2]]$param,rot=rot2)

      Z1[i] = qcondNew(Z2[i],V0,family = mod$copula[[1]]$family,cpar=mod$copula[[1]]$param,rot=rot1)
      }
      Z=c(Z,Z1)
    }
    Z = matrix(Z,ncol=p,byrow=FALSE)
    Utilde=rbind(Utilde,Ztilde)
    U=rbind(U,Z)
    V = c(V,rep(V0,nn))
    clu=c(clu,rep(k,nn))
  }

  for(j in 1:d){
    modY   = model$Y[[j]]$margin
    param  = modY$param
    family = modY$family
    # NOTE (fix): `if(cond){x=0}` with no `else` returns NULL when cond is
    # FALSE, which silently discarded any explicitly-specified size/shape/dfM
    # and passed NULL to qmargin() instead. Now falls back to the provided
    # value when present, and to 0 only when it is genuinely absent.
    size   = if(is.null(modY$size)) 0 else modY$size
    shape  = if(is.null(modY$shape)) 0 else modY$shape
    dfM    = if(is.null(modY$dfM))   0 else modY$dfM
    Y[,j]=qmargin(Utilde[,j],family,param,size,shape,dfM)

  }


  for(j in 1:p){
    modX   = model$X[[j]]$margin
    param  = modX$param
    family = modX$family
    # NOTE (fix): same missing-else bug as above, plus these two lines
    # previously read the stale `modY` (response-margin) list left over
    # from the loop above instead of `modX` (covariate-margin) -- so any
    # covariate margin needing `shape` (e.g. Weibull) or `dfM` (Student)
    # silently got the wrong value. Both are now taken from `modX`.
    size   = if(is.null(modX$size)) 0 else modX$size
    shape  = if(is.null(modX$shape)) 0 else modX$shape
    dfM    = if(is.null(modX$dfM))   0 else modX$dfM
    X[,j]=qmargin(U[,j],family,param,size,shape,dfM)
  }


out=data.frame(Y=Y,X=X,clu=clu,V=V)
return(out)
}

