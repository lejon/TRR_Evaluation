data {
  int N; // number of observations
  int K; // number of covariates + intercept (columns in the model matrix)
  real y[N]; // array of N outcomes
  matrix[N,K] X; // model matrix
  real intercept_prior_var;
  real intercept_prior_mean;
}

parameters {
  real<lower=0> sigma;
  vector[K] betas; // regression parameters
}

model {  
  betas[1]  ~ normal(intercept_prior_mean,intercept_prior_var);
  betas[2:] ~ normal(0,1); // Following Gelman 2015 
  sigma ~ exponential(1); // prior on scale
  y ~ lognormal(X*betas,sigma);
}

generated quantities {
  real y_rep[N];
  vector[N] log_lik;
  vector[N] mu;
  mu = X*betas;
  //posterior draws to get posterior predictive checks
  
  for (n in 1:N)  {
    y_rep[n] = lognormal_rng(mu[n],sigma); 
    log_lik[n] = lognormal_lpdf(y[n] | X[n, ] * betas, sigma);
  }
}
