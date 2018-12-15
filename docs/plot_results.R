#to run, use setwd([this directory])
res_full=fromJSON('./loan_density_full.json')
res_aggr=fromJSON('./loan_density_aggr.json')

get_max_vectors=function(vec1, vec2){
  mx_vec1=max(vec1)
  mx_vec2=max(vec2)
  max(mx_vec1, mx_vec2)
}
#to save as jpg
jpeg('density_compare.jpg')
plot(res_full$x, res_full$density, type='l', 
     col='blue', xlab="Dollar Losses", ylab="Density", 
     ylim=c(0, get_max_vectors(res_aggr$density, res_full$density)*1.05),
     xlim=c(.3*min(res_aggr$x), 0)
  )
lines(res_aggr$x, res_aggr$density, col='red')

dev.off()