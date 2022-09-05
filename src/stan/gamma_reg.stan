data {
  int N; // number of observations
  int K; // number of covariates + intercept (columns in the model matrix)
  real y[N]; // array of N outcomes
  matrix[N,K] X; // model matrix
  real intercept_prior_var;
  real intercept_prior_mean;

}
parameters {
  vector[K] betas; // regression parameters
  real<lower=0> inverse_phi; // variance parameter
}
transformed parameters {
  vector[N] mu; // expected value of linear predictor
  vector[N] beta; // rate parameter for the gamma distribution
  
  mu = exp(X*betas);
  beta = rep_vector(inverse_phi, N) ./ mu;
}
model {  
  betas[1]  ~ normal(intercept_prior_mean,intercept_prior_var);
  betas[2:] ~ normal(0,1); // Following Gelman 2015
  
  inverse_phi ~ exponential(1); // prior on inverse dispersion parameter
  y ~ gamma(inverse_phi,beta);
}
generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;

  for(n in 1:N){
    //posterior draws to get posterior predictive checks
    y_rep[n] = gamma_rng(inverse_phi,beta[n]); 
    log_lik[n] = gamma_lpdf(y[n] | inverse_phi, beta);
  }
}
