##########################################################################################################################################
# Simulation study for Exact and Robust Conformal Inference Methods for Predictive Machine Learning With Dependent Data
# Authors: V. Chernozhukov, K. Wuthrich, Y. Zhu
# DISCLAIMER: This software is provided "as is" without warranty of any kind, expressed or implied. 
# Questions/error reports: kwuthrich@ucsd.edu
##########################################################################################################################################

rm(list = ls())

### Packages
library(hdm)
library(doParallel)
registerDoParallel(cores=8)


### Functions
generate.AR.series = function(T,rho){
  u = matrix(NA,T,1)
  epsl = matrix(rnorm(T),T,1)
  u[1,1] = epsl[1,]*sqrt(1-rho^2)
  for (t in 2:T){
    u[t,1] = rho*u[(t-1),]+epsl[t,]*sqrt(1-rho^2)
  }
  return(u)
}

sim = function(T01,rho,p,beta,alpha){
  X = matrix(rnorm(T01 * p), T01, p)
  eps = generate.AR.series(T01,rho)
  Y = X %*% beta + eps
  y.grid = c(seq(min(Y),max(Y),length=100))
  # Coverage
  eps.hat = rlasso(X,Y,post=FALSE,penalty = list(homoscedastic = FALSE, X.dependent.lambda = FALSE))$residuals
  cov = (mean(abs(eps.hat)>=abs(eps.hat[T01])) > alpha)
  # Length
  p.vec = matrix(NA,length(y.grid),1)
  for (i in 1:length(y.grid)){
    Y[T01] = y.grid[i]
    eps.hat = rlasso(X,Y,post=FALSE,penalty = list(homoscedastic = FALSE, X.dependent.lambda = FALSE))$residuals
    p.vec[i,1] = mean(abs(eps.hat)>=abs(eps.hat[T01]))
  }
  ci = y.grid[p.vec>alpha]
  leng = max(ci)-min(ci)
  return(c(cov,leng))
}

### Simulation
set.seed(123, kind = "L'Ecuyer-CMRG") 

nsim = 2000

alpha = 0.1
rho.vec = c(seq(0,0.8,0.2),0.95)

T01.vec = c(100,200)
p = 100

beta = c(rep(1, 5), rep(0, p - 5))
beta = 2 * beta/sqrt(sum(beta^2))

results.cov = results.leng = matrix(NA,length(rho.vec),length(T01.vec))
a = Sys.time()
for (t in 1:length(T01.vec)){
  T01 = T01.vec[t]
  for (r in 1:length(rho.vec)){
    rho =  rho.vec[r]
    results.par = foreach(i=1:nsim) %dopar% {
      sim(T01,rho,p,beta,alpha)
    }
    results = rowMeans(matrix(unlist(results.par),2,nsim))
    results.cov[r,t] = results[1]
    results.leng[r,t] = results[2]
  }
}
Sys.time()-a
stopImplicitCluster()


v.axis = c(seq(0,1,0.1))

pdf("/Users/kasparwuthrich/Dropbox/research/SC/SC with Victor and Yinchu/COLT Paper/coverage_009.pdf", pointsize=15,width=8.0,height=8.0)
plot(c(0,1),c(0,1), type="n", xlab=expression(rho) , ylab="", main="")
title(main= "Coverage")
axis(side = 1, at = v.axis)
axis(side = 2, at = v.axis)
abline(a = 0.9,b=0,lty=1,col = "grey")
par(lty = 2, col = 1, lwd = 2)
lines(rho.vec, results.cov[,1])
par(lty = 3, col = 1, lwd = 2)
lines(rho.vec, results.cov[,2])
par(lty = 1, lwd = 1, col = 1)
legend("bottomleft", c("T=100", "T=200"), lty = c(2, 3), lwd = c(2,2), col = c(1,1), bty = "n")
dev.off()

pdf("/Users/kasparwuthrich/Dropbox/research/SC/SC with Victor and Yinchu/COLT Paper/length_009.pdf", pointsize=15,width=8.0,height=8.0)
plot(c(0,1),c(0,5), type="n", xlab=expression(rho), ylab="", main="")
title(main= "Length")
axis(side = 1, at = v.axis)
par(lty = 2, col = 1, lwd = 2)
lines(rho.vec, results.leng[,1])
par(lty = 3, col = 1, lwd = 2)
lines(rho.vec, results.leng[,2])
par(lty = 1, lwd = 1, col = 1)
legend("bottomleft", c("T=100", "T=200"), lty = c(2, 3), lwd = c(2,2), col = c(1,1), bty = "n")
dev.off()






