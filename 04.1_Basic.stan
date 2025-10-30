functions{
  //convolution function, using the fast fourier transform for efficiency.
  //some calculations are performed in log space for numerical stability.
  //a 64 bit system shouldn't have issues with the values I encountered, but for some reason mine does.
  vector convolution(real mu, real phi, int N, vector R){
    //calculate the unnormalized impulse response function values for each time step.
    //this parameterization of the negative binomial distribution uses a mean, mu, and dispersion parameter, phi.
    vector[N] log_IRF_unnormalized;
      for(i in 1:N){
        log_IRF_unnormalized[i] = neg_binomial_2_lpmf(i|mu, phi);
      }
    //calculate the normalization constant.
    real log_sum_IRF = log_sum_exp(log_IRF_unnormalized);
    
    //calculate the normalized IRF values for each time step. 
    //This is equivalent to dividing by the sum.
    vector[N] log_IRF = log_IRF_unnormalized - log_sum_IRF;
    
    //Perform the convolution in the frequency domain using the fast Fourier transform.
    complex_vector[N] inv_fft_product = inv_fft(elt_multiply(fft(exp(log_IRF)), fft(R)));
    
    //Initialise a vector to store the output values.
    vector[N] output;
    
    //Extract the real component of the inverse transform to y.
    for(i in 1:N){
      output[i] = get_real(inv_fft_product[i]);
    }
    return(output);
  }
}

data {
 int N;                    //discrete variable of sample size 
 vector<lower=0>[N] S;     //vector of streamflow for each time step.
 vector<lower=0>[N] R;     //vector for rainfall for each time step.
}

parameters {  
  //mu, phi, betaW, betaT, beta0, and kappa defined as unconstrained reals in log space. When used later, they are exponentiated.
  //This allows very small positive numbers to be sampled as large negative numbers, which makes the geometry of posterior space easier to navigate 
  //Very small numbers otherwise have very small partial derivatives, which manifests as flat posterior space.
  
  //IRF terms, mean and dispersion of the distribution, respectively.
  real<lower=0> mu;  
  real<lower=0> phi;
  
  //the error terms for process and observation equations, respecitvely. These are standard deviations.
  real<lower=0> sigma_obs; 
}

transformed parameters{
  //calculate the effective recharge, according to catchment area and input proportion.
  vector[N] input_R = R*2.46e6; 
  
  //predicted streamflow for each time-step, based on the convolution and the effective rainfall for that timestep.
  vector[N] y = convolution(mu, phi, N, input_R);
}

model {
  //priors for params. which are likely large.
  sigma_obs ~ normal(10e6,10e4);
  
  //priors for parameters IRF shape parameters.
  mu ~ cauchy(0,2.5);
  phi ~ cauchy(0,2.5);
  
  //observation equation.
  //calculated as predicted streamflow based on rain, plus the minimum flow observed during the driest season - this is a presumed constant contribution of springs to the system.
  S ~ normal(y+1.27e6, sigma_obs);
}
