data {
 int N;                    //discrete variable of sample size 
 vector<lower=0>[N] S;     //vector of streamflow for each time step.
 vector<lower=0>[N] R;     //vector for rainfall for each time step.
 vector<lower=0>[N] wind;  //vector for average windspeed for each time step.
}

parameters {  
  //IRF terms, mean and dispersion of the distribution, respectively.
  vector<lower=0>[N] mu;  
  vector<lower=0>[N] phi;
  
  //scaling coefficient for the antecedent rainfal ~ effective recharge relationship.
  real kappa;
  
  //time variatn scaling term for the exponential decay of antecedent rainfall.
  vector<lower=0>[N] chi;

  
  //coefficients for the multiple linear regression for the scale term, chi, of the exponential decay function.
  real<lower=0> beta0_chi;
  real<lower=0> beta0_mu;
  real<lower=0> beta0_phi;
  real beta1_chi;
  real beta1_mu;
  real beta1_phi;

  //The volume of antecedent rain effecting recharge effectiveness each time step.
  vector<lower=0>[N] antecedent_rain;
  
  //Error terms.
  real<lower=0> sigma_ant;
  real<lower=0> sigma_chi;
  real<lower=0> sigma_mu;
  real<lower=0> sigma_phi;
  real<lower=0> sigma_obs; 
}

transformed parameters{
  //Calculate the mean effective volume of rainfall (past and present) impacting system behaviour each time-step.
  //This will be used to estimate the real, latent value, which we use to estimate IRF shape parameters and recharge effectiveness.
  //Not all of this volume will appear as streamflow, and this is not intended to be mass-balanced.
  
  //Initialise a vector to estimate the predicted antecedent rainfall effecting infiltration in each time step.
  vector[N] pred_antecedent_rain;
  
  //Initialise a real to store the accumulated a decayed past rainfall.
  real decayed_past_rain = 0.0;
  
  //Handle the first iterant separately, as there is no i-1.
  pred_antecedent_rain[1] = R[1];
  
  //Accumulate and decay past rain, then calculate the decayed + newly input rain.
  for(i in 2:N){
    decayed_past_rain = (pred_antecedent_rain[i-1])*exp(-chi[i]);
    pred_antecedent_rain[i] = R[i] + decayed_past_rain;
  }
  
  //Calculate the proportion of rain which ends up in the river for each time step. 
  
  //This is a logistic function with asymptotes of zero and one,
  //rightwards translation (-5) allows values down to ~0.005 instead of the standard 0.5 for a positive input.
  vector[N] input_prop = inv_logit(exp(kappa)*antecedent_rain-5);
  
  //calculate the effective recharge, according to catchment area and input proportion.
  vector[N] input_R = elt_multiply(R*2.46e6, input_prop); 

  //Calculate the input response function for each time-step, i.e. what proportion of input volume reaches the stream each time step.
  //assume all rain reaches the river in seven days.
  //needed to reduce computational complexity to 7N from 0.5N^2

  
  //initialise a zero matrix.
  matrix[N, N] IRF = rep_matrix(0, N, N);
  
  for(i in 1:N){
    //initialise a seven element row vector.
    row_vector[7] log_IRF_unnormalized;
    
    for(j in 1:7){
      //calculate the unnormalized log IRF values.
      log_IRF_unnormalized[j] = neg_binomial_2_lpmf(j-1 | mu[i], phi[i]);
      }
    //calculate the normalization constant.
    real log_sum_IRF = log_sum_exp(log_IRF_unnormalized);
    
    //calculate the normalized IRF values for each time step. 
    //This is equivalent to dividing by the sum.
    row_vector[7] normalized_IRF = to_row_vector(exp(log_IRF_unnormalized - log_sum_IRF));

    //Add them to the appropriate position in the IRF vector.
    //end_col and len are needed to prevent index issues when i approaches N.
    int end_col = min(i+6,N);
    int len = end_col - i + 1;
    IRF[i, i : end_col] = normalized_IRF[1 : len];
  }

  //Calculate the contribution of each rainfall event to each timestep.
  //Produce a matrix where the ith column sum equals the total contributions to the ith time step, 
  //and the jth row sum equals the scaled input volume of rain to the jth time step.
  //Then take the column sums to produce a predicted value vector.
  
  //Start by initializing a zero matrix. 
  matrix[N, N] y_frag = rep_matrix(0, N, N);
  
  //Then "spread out" the volume of each rainfall event across timesteps succedeeding it, replacing the zeros in those element positions.
  //Its contribution to antecedent time-steps remains zero.
  for(i in 1:N){
    int end_col = min(i+6,N);
    y_frag[i,i:end_col] = IRF[i,i:end_col]*input_R[i];
  }
  
  //Take the column-wise sum of y_frag, abusing matrix-vector multiplication rules.
  //produces a row vector where the ith element is the sum of all rainfal contributions to the ith time step.
  //this equals the predicted streamflow for that timestep.
  row_vector[N] rv = rep_row_vector(1.0, N);
  row_vector[N] y_sum = rv * y_frag;
}

model {
  beta0_chi ~ cauchy(0,2.5);
  beta1_chi ~ cauchy(0,2.5);
  beta0_mu ~ cauchy(0,2.5);
  beta1_mu ~ cauchy(0,2.5);
  beta0_phi ~ cauchy(0,2.5);
  beta1_phi ~ cauchy(0,2.5);
  
  kappa ~ cauchy(0,2.5);
  
  sigma_obs ~ normal(10e6,10e4);
  sigma_ant ~ normal(10e6,10e4);
  sigma_chi ~ cauchy(0,2.5);
  sigma_mu ~ cauchy(0,2.5);
  sigma_phi ~ cauchy(0,2.5);
  
  //Linear regression statements to estimate paramters chi, mu, and phi.
  chi ~ normal(beta0_chi+exp(beta1_chi)*wind, sigma_chi);
  mu ~ normal(beta0_mu + exp(beta1_mu)*antecedent_rain, sigma_mu);
  phi ~ normal(beta0_phi + exp(beta1_phi)*antecedent_rain, sigma_phi);

  //Process equation for the volume of antecedent rain still impacting recharge effectiveness.
  antecedent_rain ~ normal(pred_antecedent_rain, sigma_ant);

  //Observation equation for streamflow.
  //Calculated as predicted streamflow based on rain, 
  //plus the minimum flow observed during the driest season - this is a presumed constant contribution of springs to the system.
  S ~ normal(y_sum+1.27e6, sigma_obs);
}
