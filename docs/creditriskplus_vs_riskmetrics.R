library(calibrate)
#this is actually E[X_i X_j], not the actual correlation
correlation_riskmetrics=function(p1, p2, rho){
  d1=qnorm(p1)
  d2=qnorm(p2)
  den=sqrt(1-rho*rho)
  integrate(
    function(z){
      pnorm((d1-rho*z)/den)*pnorm((d2-rho*z)/den)*dnorm(z)
    }, -Inf, Inf
  )$value
}
#this is actually E[X_i X_j], not the actual correlation. v is the standard deviation of the systemic variable
correlation_creditriskplus=function(p1, p2, v){
  p1*p2*(v*v+1)
}
#returns volatility for credit risk given "corr" (output from riskmetrics)
inverse_corr_creditriskplus=function(p1, p2, corr){
  sqrt(corr/(p1*p2)-1)
}

rho=seq(0, 1, by=.05)
p1=.03
p2=.05
standard_deviation=sapply(rho, function(rho_inst){
  corr=correlation_riskmetrics(p1, p2, rho_inst)
  inverse_corr_creditriskplus(p1, p2, corr)
})
jpeg('vol_corr_compare.jpg')
plot(rho, standard_deviation, type='l')
label_indeces=c(5, 8, 10, 14)
labels=sapply(label_indeces, function(index){
  paste("rho:", rho[index], "sd:", round(standard_deviation[index], 3))
})
textxy(rho[label_indeces], standard_deviation[label_indeces], labels)
dev.off()
