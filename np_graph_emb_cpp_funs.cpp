#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]

//[[Rcpp::export]]
Rcpp::List np_model_init_cpp(Rcpp::List processed_data, int p_len, double gamma, double init_sd = 0){
  // data
  arma::vec x_index_pos = processed_data["x_index_pos"];
  arma::mat x_index_neg = processed_data["x_index_neg"];
  arma::vec x_val_pos = processed_data["x_val_pos"];
  Rcpp::List context_index_pos = processed_data["context_index_pos"];
  context_index_pos = clone(context_index_pos);
  Rcpp::List context_val_pos = processed_data["context_val_pos"];
  context_val_pos = clone(context_val_pos);
  
  arma::vec item_index = processed_data["item_index"];
  arma::vec item_cnt =  processed_data["item_cnt"];
  
  // basic statistics
  int n_len_pos = x_index_pos.n_elem;
  int n_len_neg = x_index_neg.n_cols * x_index_neg.n_rows;
  int num_negative_samples = x_index_neg.n_cols;
  int n_len = n_len_pos + n_len_neg;
  int t_len = item_index.n_elem;
  
  Rcpp::List data = List::create(Named("x_index_pos") = x_index_pos,
                                 Named("x_index_neg") = x_index_neg,
                                 Named("x_val_pos") = x_val_pos,
                                 Named("context_index_pos") = context_index_pos,
                                 Named("context_val_pos") = context_val_pos,
                                 Named("num_negative_samples") = num_negative_samples,
                                 Named("n_len") = n_len,
                                 Named("n_len_pos") = n_len_pos,
                                 Named("n_len_neg") = n_len_neg,
                                 Named("t_len") = t_len,
                                 Named("item_index") = item_index,
                                 Named("item_cnt") = item_cnt);
  
  // hyper parameters
  arma::vec gamma_vec(t_len);
  for(int t = 0; t < t_len; ++t ){
    gamma_vec[t] = gamma;
  }
  
  Rcpp::List hy_par = List::create(Named("gamma_vec") = gamma_vec,
                                   Named("p_len") = p_len);
  
  // variational parameters
  Rcpp::List theta_mean(t_len);
  for(int t = 0; t < t_len; ++t ){
    arma::vec theta_mean_t(2);
    theta_mean_t.zeros();
    theta_mean_t[0] = item_cnt[t];
    theta_mean[t] = theta_mean_t;
  }
  
  Rcpp::List theta_var(t_len);
  for(int t = 0; t < t_len; ++t ){
    arma::vec theta_var_t(2);
    theta_var_t.zeros();
    theta_var[t] = theta_var_t;
  }
  
  Rcpp::List pi(t_len);
  for(int t = 0; t < t_len; ++t ){
    // normalize
    arma::vec pi_t =  theta_mean[t];
    double sum = arma::accu(pi_t);
    for(int i = 0; i < pi_t.n_elem; ++i ){
      pi_t[i] = pi_t[i] / sum;
    }
    // assign
    pi[t] = pi_t;
  }
  
  arma::vec s_len_vec(t_len);
  for(int t = 0; t < t_len; ++t ){
    arma::vec theta_mean_t = theta_mean[t];
    s_len_vec[t] = (theta_mean_t.n_elem - 1);
  }
  
  Rcpp::List phi_map(t_len);
  for(int t = 0; t < t_len; ++t ){
    phi_map[t] = (t+1);
  }
  
  Rcpp::Rcout << "initialize embeddings.." << std::endl;
  arma::mat phi = arma::randn(p_len, t_len) * init_sd;
  arma::mat alpha = arma::randn(t_len, p_len)* init_sd;
  
  Rcpp::List var_params = List::create(Named("theta_mean") = theta_mean, 
                                       Named("theta_var") = theta_var, 
                                       Named("pi") = pi,
                                       Named("s_len_vec") = s_len_vec,
                                       Named("phi_map") = phi_map,
                                       Named("phi") = phi, 
                                       Named("alpha") = alpha);
  
  // opt parameters
  int iter_t = 1;
  arma::mat iter_t_phi(p_len, t_len);
  iter_t_phi.ones();
  
  arma::mat m_phi(p_len, t_len);
  m_phi.zeros();
  arma::mat v_phi(p_len, t_len);
  v_phi.zeros();
  arma::mat m_hat_phi(p_len, t_len);
  m_hat_phi.zeros();
  arma::mat v_hat_phi(p_len, t_len);
  v_hat_phi.zeros();
  
  arma::mat m_alpha(t_len, p_len);
  m_alpha.zeros();
  arma::mat v_alpha(t_len, p_len);
  v_alpha.zeros();
  arma::mat m_hat_alpha(t_len, p_len);
  m_hat_alpha.zeros();
  arma::mat v_hat_alpha(t_len, p_len);
  v_hat_alpha.zeros();
  
  Rcpp::List opt_params = List::create(Named("m_phi") = m_phi, 
                                       Named("v_phi") = v_phi,
                                       Named("m_hat_phi") = m_hat_phi,
                                       Named("v_hat_phi") = v_hat_phi,
                                       Named("m_alpha") = m_alpha,
                                       Named("v_alpha") = v_alpha,
                                       Named("m_hat_alpha") = m_hat_alpha,
                                       Named("v_hat_alpha") = v_hat_alpha,
                                       Named("iter_t") = iter_t,
                                       Named("iter_t_phi") = iter_t_phi);
  
  Rcpp::List np_model = List::create(Named("data") = data,
                                     Named("hy_par") = hy_par,
                                     Named("opt_params") = opt_params, 
                                     Named("var_params") = var_params);
  
  return(np_model);
}

//[[Rcpp::export]]
Rcpp::List balanced_sampling_ccp(arma::uvec indices_m_pos, Rcpp::List data){
  int indices_len = indices_m_pos.n_elem;
  
  arma::vec x_index_pos = data["x_index_pos"];
  arma::vec x_val_pos = data["x_val_pos"];
  arma::mat x_index_neg = data["x_index_neg"];
  Rcpp::List context_index_pos = data["context_index_pos"];
  context_index_pos = clone(context_index_pos);
  Rcpp::List context_val_pos = data["context_val_pos"];
  context_val_pos = clone(context_val_pos);
  int num_negative_samples = data["num_negative_samples"];
  
  arma::vec x_index(indices_len * (1 + num_negative_samples));
  arma::vec x_val(indices_len * (1 + num_negative_samples));
  x_val.zeros();
  
  Rcpp::List context_index(indices_len * (1 + num_negative_samples));
  Rcpp::List context_val(indices_len * (1 + num_negative_samples));
  
  for(int i = 0; i < indices_len; i++){
    x_index[i] = x_index_pos[indices_m_pos[i]];
    x_val[i] = x_val_pos[indices_m_pos[i]];
    context_index[i] = context_index_pos[indices_m_pos[i]];
    context_val[i] = context_val_pos[indices_m_pos[i]];
    
    int int_start = (i * num_negative_samples) + indices_len;
    int int_end = int_start + (num_negative_samples-1);
    for(int n = int_start; n <= int_end; n++){
      x_index[n] = x_index_neg(indices_m_pos[i], (n - int_start));
      context_index[n] = context_index_pos[indices_m_pos[i]];
      context_val[n] = context_val_pos[indices_m_pos[i]];
    }
  }
  
  Rcpp::List out = List::create(Named("x_index") = x_index,
                                Named("x_val") = x_val,
                                Named("context_index") = context_index,
                                Named("context_val") = context_val);
  
  
  return(out);
}

//[[Rcpp::export]]
Rcpp::List zeta_init(int len){
  Rcpp::List zeta(len);
  for(int i = 0; i < len; ++i){
    arma::vec zeta_i(1);
    zeta_i.ones();
    zeta[i] = zeta_i;
  }
  return(zeta);
}

//[[Rcpp::export]]
Rcpp::List zeta_normalize(Rcpp::List zeta){
  int n_len = zeta.size();
  Rcpp::List zeta_out(n_len);
  int s_len;
  double denom;
  for(int n = (n_len - 1); n >=0; n--){
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    
    arma::vec zeta_out_n(s_len - 1);
    zeta_out_n.zeros();
    
    for(int s = (s_len - 2); s >=0; s--){
      zeta_out_n[s] = zeta_n[s];
    }
    denom = arma::accu(zeta_out_n);
    
    if(denom == 0){
      zeta_out_n[0] = 1;
    }else{
      for(int s = (s_len - 2); s >=0; s--){
        zeta_out_n[s] = zeta_out_n[s] / denom;
      }
    }
    zeta_out[n] = zeta_out_n;
  }
  
  return(zeta_out);
}

//[[Rcpp::export]]
arma::mat c_eff_cpp(arma::mat alpha,
                    Rcpp::List context_val,
                    Rcpp::List context_index) {
  
  int n_len = context_val.length(); // length of observed sequence
  int p_len = alpha.n_cols; // dimension of embedding vectors
  
  arma::mat out_mtx(n_len, p_len);
  out_mtx.zeros();
  
  int w_len, t; // w_len: context width, t: item indicator
  double val_context, norm_const;
  for(int n = (n_len - 1); n >=0; n--) {
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = 1/double(w_len);
    
    for(int w = (w_len - 1); w >=0; w--){
      t = context_index_n[w] - 1;
      val_context = context_val_n[w];
      
      for(int p = (p_len - 1); p >=0; p--){
        // Rcpp::Rcout << alpha(v, p) << std::endl;
        out_mtx(n,p) += alpha(t, p) * val_context * norm_const;
      }
    }
  }
  return(out_mtx);
}

//[[Rcpp::export]]
arma::mat grad_phi_norm_cpp(arma::vec x_val,
                            arma::vec x_index,
                            arma::mat phi,
                            arma::mat R,
                            Rcpp::List phi_map,
                            Rcpp::List zeta, double sigma, double sigma_prior) {
  
  int n_len = x_val.n_elem;
  int p_len = phi.n_rows;
  arma::mat grad_phi(p_len, int(phi.n_cols));
  grad_phi.zeros();
  
  // gradient of prior
  grad_phi = phi / (-1 * sigma_prior*sigma_prior);
  
  int t_center, s_len, phi_ind;
  arma::vec phi_s;
  arma::mat R_n;
  double val_center, mu, d_phi, d_mu;
  for(int n = (n_len-1); n >= 0; n--){
    t_center = int(x_index[n] - 1); //index of the item (index difference for R and c++)
    val_center = x_val[n]; //value at the n-th position
    R_n = R.row(n);
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    arma::vec phi_map_t = phi_map[t_center];
    
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      phi_ind = int(phi_map_t[s] - 1);
      phi_s = phi.col(phi_ind);
      mu = arma::accu(R_n * phi_s);
      d_mu = (val_center - mu) / (sigma * sigma);
      
      // if(n == 0){
      //   Rcpp::Rcout << "zeta_s: "<< zeta_s << std::endl;
      //   Rcpp::Rcout << "mu: "<< mu << std::endl;
      //   Rcpp::Rcout << "val_center: "<< val_center << std::endl;
      //   Rcpp::Rcout << "d_mu: "<< d_mu << std::endl;
      // }
      
      for(int p = (p_len - 1); p >=0; p--){
        d_phi = R_n[p];
        grad_phi(p, phi_ind) += (d_mu * d_phi * zeta_s);
        
        // if(n == 0 & p == 0){
        //   Rcpp::Rcout << "phi_ind: "<< phi_ind << std::endl;
        //   Rcpp::Rcout << "grad_phi(p, phi_ind): "<< grad_phi(p, phi_ind) << std::endl;
        // }
      }
    }
  }
  
  
  // grad_phi(0,0) += 1;
  // Rcpp::Rcout << "grad_phi(p, phi_ind): "<< grad_phi(0, 1237) << std::endl;
  return(grad_phi);
}


//[[Rcpp::export]]
arma::mat grad_phi_pois_cpp(arma::vec x_val,
                            arma::vec x_index,
                            arma::mat phi,
                            arma::mat R,
                            Rcpp::List phi_map,
                            Rcpp::List zeta, double sigma_prior) {
  
  int n_len = x_val.n_elem;
  int p_len = phi.n_rows;
  arma::mat grad_phi(int(phi.n_rows), int(phi.n_cols));
  grad_phi.zeros();
  
  // gradient of prior
  grad_phi = phi / (-1 * sigma_prior*sigma_prior);
  
  int t_center, val_center, s_len, phi_ind;
  arma::vec phi_s;
  arma::mat R_n;
  double vec_prod, lambda, d_lambda, d_phi;
  for(int n = (n_len-1); n >= 0; n--){
    t_center = x_index[n] - 1; //index of the item (index difference for R and c++)
    val_center = x_val[n]; //value at the n-th position
    R_n = R.row(n);
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    arma::vec phi_map_t = phi_map[t_center];
    
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      // if(n == 0){
      //   Rcpp::Rcout << "s:" << s << std::endl;
      // }
      // if(n == 0){
      //   Rcpp::Rcout << "zeta_s:" << zeta_s << std::endl;
      // }
      phi_ind = int(phi_map_t[s] - 1);
      phi_s = phi.col(phi_ind);
      lambda = exp(arma::accu(R_n * phi_s));
      
      d_lambda = ((val_center / lambda) - 1);
      
      for(int p = (p_len - 1); p >=0; p--){
        d_phi = R_n[p] * lambda;
        grad_phi(p, phi_ind) += (d_lambda * d_phi * zeta_s);
      }
    }
  }
  return(grad_phi);
}

//[[Rcpp::export]]
arma::mat grad_alpha_norm_cpp(arma::vec x_val,
                              arma::vec x_index,
                              Rcpp::List context_val,
                              Rcpp::List context_index,
                              arma::mat phi,
                              arma::mat alpha,
                              arma::mat R,
                              Rcpp::List phi_map,
                              Rcpp::List zeta,
                              int t_len,
                              double sigma, double sigma_prior){
  
  int n_len = x_index.n_elem;
  int p_len = phi.n_rows;
  
  arma::mat out_grad(t_len, p_len);
  out_grad.zeros();
  
  // gradient of prior
  out_grad = alpha/(-1*sigma_prior*sigma_prior);
  
  int w_len, t_center, t_context, s_len, phi_ind;
  arma::mat R_n; 
  arma::vec phi_s;
  double val_center, val_context, vec_prod, mu, d_mu, d_alpha, norm_const;
  
  for(int n = (n_len-1); n >=0; n--){
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = (1/double(w_len));
    
    t_center = int(x_index[n]-1);
    val_center = x_val[n];
    R_n = R.row(n);
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    arma::vec phi_map_t = phi_map[t_center];
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      phi_ind = int(phi_map_t[s] - 1);
      phi_s = phi.col(phi_ind);
      
      mu = arma::accu(R_n * phi_s);
      d_mu = (val_center - mu) / (sigma * sigma);
      
      // if(n == 0){
      //   Rcpp::Rcout << "mu: "<< mu << std::endl;
      //   Rcpp::Rcout << "val_center: "<< val_center << std::endl;
      //   Rcpp::Rcout << "d_mu: "<< d_mu << std::endl;
      //   Rcpp::Rcout << "sigma: "<< sigma << std::endl;
      // }
      
      for(int w = (w_len-1); w >=0; w--){
        t_context = int(context_index_n[w]-1);
        val_context = context_val_n[w];
        
        for(int p = (p_len - 1); p >=0; p--){
          // d_mu = norm_const * phi_s[p] * zeta_s;
          d_alpha = norm_const * phi_s[p] * val_context;
          // if(n == 0 & p == 0){
          //   Rcpp::Rcout << "d_alpha: "<< d_alpha << std::endl;
          // }
          
          out_grad(t_context, p) += (d_mu * d_alpha * zeta_s);
        }
      }
    }
  }
  
  return(out_grad) ;
}

//[[Rcpp::export]]
arma::mat grad_alpha_pois_cpp(arma::vec x_val,
                              arma::vec x_index,
                              Rcpp::List context_val,
                              Rcpp::List context_index,
                              arma::mat phi,
                              arma::mat alpha,
                              arma::mat R,
                              Rcpp::List phi_map,
                              Rcpp::List zeta,
                              int t_len, double sigma_prior){
  
  int n_len = x_index.n_elem;
  int p_len = phi.n_rows;
  
  arma::mat out_grad(t_len, p_len);
  out_grad.zeros();
  
  // gradient of prior
  out_grad = alpha/(-1*sigma_prior*sigma_prior);
  
  int w_len, t_center, val_center, t_context, val_context, s_len, phi_ind;
  arma::mat R_n; 
  arma::vec phi_s;
  double vec_prod, lambda, d_lambda, d_alpha, norm_const;
  
  for(int n = (n_len-1); n >=0; n--){
    arma::vec context_val_n = context_val[n];
    arma::vec context_index_n = context_index[n];
    w_len = context_val_n.n_elem;
    norm_const = (1/double(w_len));
    
    t_center = x_index[n]-1;
    val_center = x_val[n];
    R_n = R.row(n);
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    arma::vec phi_map_t = phi_map[t_center];
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      phi_ind = int(phi_map_t[s] - 1);
      phi_s = phi.col(phi_ind);
      
      lambda = exp(arma::accu(R_n * phi_s));
      d_lambda = (val_center / lambda) - 1;
      
      for(int w = (w_len-1); w >=0; w--){
        t_context = context_index_n[w]-1;
        val_context = context_val_n[w];
        
        for(int p = (p_len - 1); p >=0; p--){
          d_alpha = norm_const * phi_s[p] * val_context * lambda * zeta_s;
          out_grad(t_context, p) = out_grad(t_context, p) + (d_lambda * d_alpha);
        }
      }
    }
  }
  
  return(out_grad) ;
}

//[[Rcpp::export]]  
Rcpp::List grad_emb_norm_stoch_cpp(Rcpp::List samples, 
                                   arma::mat phi,
                                   arma::mat alpha,
                                   Rcpp::List phi_map,
                                   Rcpp::List zeta, 
                                   double sigma, 
                                   int t_len, double sigma_prior){
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  // arma::mat phi = var_params["phi"];
  // arma::mat alpha = var_params["alpha"];
  // Rcpp::List phi_map = var_params["phi_map"];
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  arma::mat phi_grad = grad_phi_norm_cpp(x_val,
                                         x_index,
                                         phi,
                                         R,
                                         phi_map,
                                         zeta, sigma, sigma_prior);
  
  arma::mat alpha_grad = grad_alpha_norm_cpp(x_val,
                                             x_index,
                                             context_val,
                                             context_index,
                                             phi,
                                             alpha,
                                             R,
                                             phi_map,
                                             zeta,
                                             t_len,
                                             sigma, sigma_prior);
  
  Rcpp::List out = List::create(Named("phi_grad") = phi_grad,
                                Named("alpha_grad") = alpha_grad);
  return(out);
}


//[[Rcpp::export]]  
Rcpp::List grad_emb_pois_stoch_cpp(Rcpp::List samples, 
                                   arma::mat phi,
                                   arma::mat alpha,
                                   Rcpp::List phi_map,
                                   Rcpp::List zeta, 
                                   int t_len, double sigma_prior){
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  // arma::mat phi = var_params["phi"];
  // arma::mat alpha = var_params["alpha"];
  // Rcpp::List phi_map = var_params["phi_map"];
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  arma::mat phi_grad = grad_phi_pois_cpp(x_val,
                                         x_index,
                                         phi,
                                         R,
                                         phi_map,
                                         zeta, sigma_prior);
  
  arma::mat alpha_grad = grad_alpha_pois_cpp(x_val,
                                             x_index,
                                             context_val,
                                             context_index,
                                             phi,
                                             alpha,
                                             R,
                                             phi_map,
                                             zeta,
                                             t_len, sigma_prior);
  
  Rcpp::List out = List::create(Named("phi_grad") = phi_grad,
                                Named("alpha_grad") = alpha_grad);
  return(out);
}

//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_lkhd_norm_cpp(arma::vec x_val,
                                      arma::vec x_index,
                                      arma::mat phi,
                                      arma::mat R,
                                      Rcpp::List phi_map,
                                      arma::vec s_len_vec,
                                      double sigma) {
  
  // initialize output vector
  int n_len = x_val.n_elem;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s;
  double val_center, mu, denom, item_cnt_t;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    t_center = int(x_index[n] - 1);
    s_len = s_len_vec[t_center];
    arma::vec zeta_n((s_len));
    zeta_n.zeros();
    
    if(s_len == 1){
      zeta_n[0] = 1;
    }else{
      R_n = R.row(n);
      val_center = x_val[n];
      arma::vec phi_map_t = phi_map[t_center];
      
      denom = 0;
      for(int s = (s_len - 1); s >=0; s--) {
        phi_ind = (phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        mu = arma::accu(R_n * phi_s);
        
        //likelihood
        zeta_n[s] += R::dnorm(val_center, mu, sigma, 1);
        
        // if(n == 1){
        //   Rcpp::Rcout << s << std::endl;
        //   Rcpp::Rcout << zeta_n[s] << std::endl;
        //   Rcpp::Rcout << phi_ind << std::endl;
        //   Rcpp::Rcout << mu << std::endl;
        // }
        
        denom += exp(zeta_n[s]);
      }
      
      if(denom == 0 || denom == R_PosInf){
        denom = 0;
        // double threshold = Rcpp::max(zeta_n);
        double threshold = arma::max(zeta_n);
        // Rcpp::Rcout << threshold << std::endl;
        for(int s = (s_len - 1); s >=0; s--) {
          zeta_n[s] -= threshold;
          denom += exp(zeta_n[s]);
        }
      }
      
      // if(n == 16){
      //   Rcpp::Rcout << zeta_n << std::endl;
      //   Rcpp::Rcout << denom << std::endl;
      // }
      
      // normalization
      for(int s = s_len; s >=0; s--) {
        zeta_n[s] = exp(zeta_n[s]) / denom;
      }
    }
    
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}

//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_lkhd_pois_cpp(arma::vec x_val,
                                      arma::vec x_index,
                                      arma::mat phi,
                                      arma::mat R,
                                      Rcpp::List phi_map,
                                      arma::vec s_len_vec) {
  
  // initialize output vector
  int n_len = x_val.n_elem;
  // Rcpp::Rcout << "n_len" << ":" << n_len << std::endl;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s, pi_vt;
  double val_center, lambda, denom;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    // Rcpp::Rcout << "n" << ":" << n << std::endl;
    t_center = int(x_index[n] - 1);
    // Rcpp::Rcout << n << ":" << t_center << std::endl;
    s_len = s_len_vec[t_center];
    arma::vec zeta_n((s_len));
    zeta_n.zeros();
    
    if(s_len == 1){
      zeta_n[0] = 1;
    }else{
      R_n = R.row(n);
      val_center = x_val[n];
      arma::vec phi_map_t = phi_map[t_center];
      
      denom = 0;
      for(int s = (s_len - 1); s >=0; s--) {
        phi_ind = int(phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        lambda = exp(arma::accu(R_n * phi_s));
        
        //likelihood
        zeta_n[s] += R::dpois(val_center, lambda, 1);
        
        // if(n == 0){
        //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
        // }
        denom += exp(zeta_n[s]);
      }
      
      if(denom == 0 || denom == R_PosInf){
        denom = 0;
        // double threshold = arma::max(zeta_n);
        double threshold = arma::max(zeta_n);
        // Rcpp::Rcout << threshold << std::endl;
        for(int s = s_len; s >=0; s--) {
          zeta_n[s] -= threshold;
          denom += exp(zeta_n[s]);
        }
      }
      
      // normalization
      for(int s = (s_len-1); s >=0; s--) {
        zeta_n[s] = exp(zeta_n[s]) / denom;
      }
    }
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}

//[[Rcpp::export]]
double log_lklh_no_prior_norm_cpp(arma::vec x_val,
                                  arma::vec x_index,
                                  arma::mat R,
                                  arma::mat phi,
                                  Rcpp::List phi_map,
                                  Rcpp::List zeta, 
                                  double sigma) {
  double out_val = 0;
  
  int n_len = x_index.size();
  int s_len, t_center, phi_ind;
  double val, pi_ts, mu;
  arma::mat R_n, phi_s;
  
  for(int n = (n_len-1); n >=0; n--) {
    // Rcpp::Rcout << "n: " << n << std::endl;
    
    t_center = int(x_index[n]-1);
    val = x_val[n];
    R_n = R.row(n);
    arma::vec phi_map_t = phi_map[t_center];
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      if(zeta_s > 0){
        phi_ind = int(phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        mu = arma::accu(R_n * phi_s);
        
        out_val = out_val + zeta_s * R::dnorm(val, mu, sigma, 1);
      }
    }
  }
  
  return(out_val);
}

//[[Rcpp::export]]
double log_lklh_no_prior_pois_cpp(arma::vec x_val,
                                  arma::vec x_index,
                                  arma::mat R,
                                  arma::mat phi,
                                  Rcpp::List phi_map,
                                  Rcpp::List zeta) {
  double out_val = 0;
  int n_len = x_index.size(); // length of observed sequence
  int t_center, s_len, phi_ind; 
  // s_len: number of embeddings
  // t: item indicator
  // phi_ind: embedding vector indicator
  
  double val, pi_ts, lambda;
  arma::mat R_n, phi_s;
  
  for(int n = (n_len-1); n >=0; n--) {
    t_center = (x_index[n]-1);
    // Rcpp::Rcout << "t_center" << ":" << t_center << std::endl;
    val = x_val[n];
    // Rcpp::Rcout << "val" << ":" << val << std::endl;
    R_n = R.row(n);
    
    arma::vec phi_map_t = phi_map[t_center];
    // Rcpp::Rcout << "phi_map_t" << ":" << phi_map_t << std::endl;
    arma::vec zeta_n = zeta[n];
    // Rcpp::Rcout << "zeta_n" << ":" << zeta_n << std::endl;
    s_len = zeta_n.n_elem;
    
    for(int s = (s_len -1); s >= 0; s--){
      double zeta_s = zeta_n[s];
      if(zeta_s > 0){
        
        phi_ind = int(phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        lambda = exp(arma::accu(R_n * phi_s));
        // Rcpp::Rcout << "lambda" << ":" << lambda << std::endl;
        
        // if(n == 0){
        //   Rcpp::Rcout << "s:" << s << std::endl;
        // }
        
        out_val = out_val + zeta_s * R::dpois(val, lambda, 1);
        // Rcpp::Rcout << "zeta_s" << ":" << zeta_s << std::endl;
        // Rcpp::Rcout << "R::dpois(val, lambda, 1)" << ":" << R::dpois(val, lambda, 1) << std::endl;
        // Rcpp::Rcout << "out_val" << ":" << out_val << std::endl;
      }
    }
  }
  
  return(out_val);
}


//[[Rcpp::export]]
double tracking_lkhd_norm_cpp(Rcpp::List track_data,
                              arma::mat phi,
                              arma::mat alpha,
                              Rcpp::List phi_map,
                              arma::vec s_len_vec,
                              double sigma){
  
  arma::vec x_val = track_data["x_val"];
  arma::vec x_index = track_data["x_index"];
  Rcpp::List context_val = track_data["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = track_data["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  Rcpp::List zeta_obs = zeta_upd_vec_lkhd_norm_cpp(x_val, x_index, phi, R, phi_map, s_len_vec, sigma);
  double lkhd_val = log_lklh_no_prior_norm_cpp(x_val, x_index, R, phi, phi_map, zeta_obs, sigma);
  return(lkhd_val);
}


//[[Rcpp::export]]
double tracking_lkhd_pois_cpp(Rcpp::List track_data,
                              arma::mat phi,
                              arma::mat alpha,
                              Rcpp::List phi_map,
                              arma::vec s_len_vec){
  
  arma::vec x_val = track_data["x_val"];
  arma::vec x_index = track_data["x_index"];
  Rcpp::List context_val = track_data["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = track_data["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  Rcpp::List zeta_obs = zeta_upd_vec_lkhd_pois_cpp(x_val, x_index, phi, R, phi_map, s_len_vec);
  double lkhd_val = log_lklh_no_prior_pois_cpp(x_val, x_index, R, phi, phi_map, zeta_obs);
  return(lkhd_val);
}

//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_up_norm_cpp(Rcpp::List samples,
                                    arma::mat phi,
                                    arma::mat alpha,
                                    Rcpp::List phi_map,
                                    arma::vec s_len_vec,
                                    arma::vec gamma_vec,
                                    double sigma) {
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  // initialize output vector
  int n_len = x_val.n_elem;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s, pi_vt;
  double val_center, mu, denom, item_cnt_t, theta_mean_minus_t_s, theta_var_minus_t_s, gamma_t;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    t_center = int(x_index[n] - 1);
    s_len = s_len_vec[t_center];
    // NumericVector zeta_n((s_len + 1));
    arma::vec zeta_n((s_len + 1));
    zeta_n.zeros();
    
    R_n = R.row(n);
    val_center = x_val[n];
    arma::vec phi_map_t = phi_map[t_center];
    
    gamma_t = gamma_vec[t_center];
    
    denom = 0;
    for(int s = s_len; s >=0; s--) {
      if(s == s_len){
        zeta_n[s] += log(gamma_t / (s_len + gamma_t));
      }else if(s < s_len){
        phi_ind = (phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        mu = arma::accu(R_n * phi_s);
        
        //likelihood
        zeta_n[s] += R::dnorm(val_center, mu, sigma, 1);
        
        // if(n == 0){
        //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
        // }
        
        // prior
        
        
        // if(n == 0){
        //   Rcpp::Rcout << "theta_mean_t_s: " << theta_mean_t[s] << std::endl;
        //   Rcpp::Rcout << "theta_var_t_s: " << theta_var_t[s] << std::endl;
        //   Rcpp::Rcout << "item_cnt_t: " << item_cnt_t << std::endl;
        //   
        //   Rcpp::Rcout << "theta_mean_minus_t_s: " << theta_mean_minus_t_s << std::endl;
        //   Rcpp::Rcout << "theta_var_minus_t_s: " << theta_var_minus_t_s << std::endl;
        //   
        //   Rcpp::Rcout << "plus: " << log((theta_mean_minus_t_s / (item_cnt_t - 1 + gamma_t))) << std::endl;
        // }
        zeta_n[s] += log(1 / (s_len + gamma_t));
      }
      
      // if(n == 0){
      //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
      // }
      
      denom += exp(zeta_n[s]);
    }
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << ":" << zeta_n << std::endl;
    // }
    
    if(denom == 0 || denom == R_PosInf){
      denom = 0;
      // double threshold = Rcpp::max(zeta_n);
      double threshold = arma::max(zeta_n);
      // Rcpp::Rcout << threshold << std::endl;
      for(int s = s_len; s >=0; s--) {
        zeta_n[s] -= threshold;
        denom += exp(zeta_n[s]);
      }
    }
    
    // normalization
    for(int s = s_len; s >=0; s--) {
      zeta_n[s] = exp(zeta_n[s]) / denom;
    }
    
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}


//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_dp_norm_cpp(Rcpp::List samples,
                                    arma::mat phi,
                                    arma::mat alpha,
                                    Rcpp::List phi_map,
                                    arma::vec s_len_vec,
                                    Rcpp::List theta_mean,
                                    Rcpp::List theta_var,
                                    arma::vec item_cnt_vec,
                                    arma::vec gamma_vec,
                                    double sigma) {
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  // initialize output vector
  int n_len = x_val.n_elem;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s;
  double val_center, mu, denom, item_cnt_t, theta_mean_minus_t_s, theta_var_minus_t_s, gamma_t;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    t_center = int(x_index[n] - 1);
    
    R_n = R.row(n);
    val_center = x_val[n];
    arma::vec phi_map_t = phi_map[t_center];
    s_len = s_len_vec[t_center];
    item_cnt_t = item_cnt_vec[t_center];
    gamma_t = gamma_vec[t_center];
    
    arma::vec theta_mean_t = theta_mean[t_center];
    arma::vec theta_var_t = theta_var[t_center];
    
    // NumericVector zeta_n((s_len + 1));
    
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << std::endl;
    // }
    
    arma::vec zeta_n((s_len + 1));
    zeta_n.zeros();
    
    denom = 0;
    for(int s = s_len; s >=0; s--) {
      if(s == s_len){
        zeta_n[s] += log(gamma_t / (item_cnt_t - 1 + gamma_t));
        
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        
      }else if(s < s_len){
        phi_ind = (phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        mu = arma::accu(R_n * phi_s);
        
        //likelihood
        zeta_n[s] += R::dnorm(val_center, mu, sigma, 1);
        
        // prior
        theta_mean_minus_t_s = theta_mean_t[s] * ((item_cnt_t - 1)/item_cnt_t);
        theta_var_minus_t_s = theta_var_t[s] * ((item_cnt_t - 1)/item_cnt_t);
        
        
        zeta_n[s] += log(theta_mean_minus_t_s / (item_cnt_t - 1 + gamma_t));
        zeta_n[s] -= 0.5 * (theta_var_minus_t_s / (theta_mean_minus_t_s * theta_mean_minus_t_s));
        
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "likelihood: " << R::dnorm(val_center, mu, sigma, 1) << std::endl;
        //   Rcpp::Rcout << "theta_mean_t_s: " << theta_mean_t[s] << std::endl;
        //   Rcpp::Rcout << "theta_var_t_s: " << theta_var_t[s] << std::endl;
        //   Rcpp::Rcout << "item_cnt_t: " << item_cnt_t << std::endl;
        //   
        //   Rcpp::Rcout << "theta_mean_minus_t_s: " << theta_mean_minus_t_s << std::endl;
        //   Rcpp::Rcout << "theta_var_minus_t_s: " << theta_var_minus_t_s << std::endl;
        //   
        //   Rcpp::Rcout << "plus: " << log((theta_mean_minus_t_s / (item_cnt_t - 1 + gamma_t))) << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
      }
      
      // if(n == 0){
      //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
      // }
      
      denom += exp(zeta_n[s]);
    }
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << ":" << zeta_n << std::endl;
    // }
    
    if(denom == 0 || denom == R_PosInf){
      denom = 0;
      // double threshold = Rcpp::max(zeta_n);
      double threshold = arma::max(zeta_n);
      // Rcpp::Rcout << threshold << std::endl;
      for(int s = s_len; s >=0; s--) {
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        zeta_n[s] -= threshold;
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        denom += exp(zeta_n[s]);
      }
    }
    
    // normalization
    for(int s = s_len; s >=0; s--) {
      zeta_n[s] = exp(zeta_n[s]) / denom;
      // if(n == 671){
      //   Rcpp::Rcout << "s: " << s << std::endl;
      //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
      // }
    }
    
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}


//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_up_pois_cpp(Rcpp::List samples,
                                    arma::mat phi,
                                    arma::mat alpha,
                                    Rcpp::List phi_map,
                                    arma::vec s_len_vec,
                                    arma::vec gamma_vec) {
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  // initialize output vector
  int n_len = x_val.n_elem;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s;
  double val_center, lambda, denom, item_cnt_t, theta_mean_minus_t_s, theta_var_minus_t_s, gamma_t;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    t_center = int(x_index[n] - 1);
    
    R_n = R.row(n);
    val_center = x_val[n];
    arma::vec phi_map_t = phi_map[t_center];
    s_len = s_len_vec[t_center];
    gamma_t = gamma_vec[t_center];
    
    // NumericVector zeta_n((s_len + 1));
    
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << std::endl;
    // }
    
    arma::vec zeta_n((s_len + 1));
    zeta_n.zeros();
    
    denom = 0;
    for(int s = s_len; s >=0; s--) {
      if(s == s_len){
        zeta_n[s] += log(gamma_t / (s_len + gamma_t));
        // if(n == 0){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        
      }else if(s < s_len){
        phi_ind = (phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        lambda = exp(arma::accu(R_n * phi_s));
        //likelihood
        zeta_n[s] += R::dpois(val_center, lambda, 1);
        // if(n == 0){
        //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
        // }
        
        // prior
        zeta_n[s] += log(1 / (s_len + gamma_t));
        // if(n == 0){
        //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
        // }
      }
      
      // if(n == 0){
      //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
      // }
      
      denom += exp(zeta_n[s]);
    }
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << ":" << zeta_n << std::endl;
    // }
    
    if(denom == 0 || denom == R_PosInf){
      denom = 0;
      // double threshold = Rcpp::max(zeta_n);
      double threshold = arma::max(zeta_n);
      // Rcpp::Rcout << threshold << std::endl;
      for(int s = s_len; s >=0; s--) {
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        zeta_n[s] -= threshold;
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        denom += exp(zeta_n[s]);
      }
    }
    
    // normalization
    for(int s = s_len; s >=0; s--) {
      zeta_n[s] = exp(zeta_n[s]) / denom;
      // if(n == 671){
      //   Rcpp::Rcout << "s: " << s << std::endl;
      //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
      // }
    }
    
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}


//[[Rcpp::export]]
Rcpp::List zeta_upd_vec_dp_pois_cpp(Rcpp::List samples,
                                    arma::mat phi,
                                    arma::mat alpha,
                                    Rcpp::List phi_map,
                                    arma::vec s_len_vec,
                                    Rcpp::List theta_mean,
                                    Rcpp::List theta_var,
                                    arma::vec item_cnt_vec,
                                    arma::vec gamma_vec) {
  
  arma::vec x_val = samples["x_val"];
  arma::vec x_index = samples["x_index"];
  Rcpp::List context_val = samples["context_val"];
  context_val = clone(context_val);
  Rcpp::List context_index = samples["context_index"];
  context_index = clone(context_index);
  
  arma::mat R = c_eff_cpp(alpha,
                          context_val,
                          context_index);
  
  // initialize output vector
  int n_len = x_val.n_elem;
  Rcpp::List out_list(n_len);
  
  int s_len, phi_ind, t_center;
  arma::mat R_n, phi_s;
  double val_center, lambda, denom, item_cnt_t, theta_mean_minus_t_s, theta_var_minus_t_s, gamma_t;
  
  for(int n = (n_len - 1); n >= 0; n--) {
    t_center = int(x_index[n] - 1);
    
    // Rcpp::Rcout << t_center << std::endl;
    
    R_n = R.row(n);
    val_center = x_val[n];
    arma::vec phi_map_t = phi_map[t_center];
    s_len = s_len_vec[t_center];
    item_cnt_t = item_cnt_vec[t_center];
    gamma_t = gamma_vec[t_center];
    
    arma::vec theta_mean_t = theta_mean[t_center];
    arma::vec theta_var_t = theta_var[t_center];
    
    // NumericVector zeta_n((s_len + 1));
    
    // if(n == 32){
    //   Rcpp::Rcout << zeta_n << std::endl;
    // }
    
    arma::vec zeta_n((s_len + 1));
    zeta_n.zeros();
    
    denom = 0;
    for(int s = s_len; s >=0; s--) {
      if(s == s_len){
        zeta_n[s] += log(gamma_t / (item_cnt_t - 1 + gamma_t));
        // if(n == 32){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        // 
      }else if(s < s_len){
        phi_ind = (phi_map_t[s] - 1);
        phi_s = phi.col(phi_ind);
        lambda = exp(arma::accu(R_n * phi_s));
        //likelihood
        zeta_n[s] += R::dpois(val_center, lambda, 1);
        
        // prior
        theta_mean_minus_t_s = theta_mean_t[s] * ((item_cnt_t - 1)/item_cnt_t);
        theta_var_minus_t_s = theta_var_t[s] * ((item_cnt_t - 1)/item_cnt_t);
        
        
        zeta_n[s] += log(theta_mean_minus_t_s / (item_cnt_t - 1 + gamma_t));
        zeta_n[s] -= 0.5 * (theta_var_minus_t_s / (theta_mean_minus_t_s * theta_mean_minus_t_s));
        
        // if(n == 32){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "likelihood: " << R::dpois(val_center, lambda, 1) << std::endl;
        //   Rcpp::Rcout << "theta_mean_t_s: " << theta_mean_t[s] << std::endl;
        //   Rcpp::Rcout << "theta_var_t_s: " << theta_var_t[s] << std::endl;
        //   Rcpp::Rcout << "item_cnt_t: " << item_cnt_t << std::endl;
        // 
        //   Rcpp::Rcout << "theta_mean_minus_t_s: " << theta_mean_minus_t_s << std::endl;
        //   Rcpp::Rcout << "theta_var_minus_t_s: " << theta_var_minus_t_s << std::endl;
        // 
        //   Rcpp::Rcout << "plus: " << log((theta_mean_minus_t_s / (item_cnt_t - 1 + gamma_t))) << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
      }
      
      // if(n == 0){
      //   Rcpp::Rcout << s << ":" << zeta_n[s] << std::endl;
      // }
      
      denom += exp(zeta_n[s]);
    }
    // if(n == 0){
    //   Rcpp::Rcout << zeta_n << ":" << zeta_n << std::endl;
    // }
    
    if(denom == 0 || denom == R_PosInf){
      denom = 0;
      // double threshold = Rcpp::max(zeta_n);
      double threshold = arma::max(zeta_n);
      // Rcpp::Rcout << threshold << std::endl;
      for(int s = s_len; s >=0; s--) {
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        zeta_n[s] -= threshold;
        // if(n == 671){
        //   Rcpp::Rcout << "s: " << s << std::endl;
        //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
        // }
        denom += exp(zeta_n[s]);
      }
    }
    
    // normalization
    for(int s = s_len; s >=0; s--) {
      zeta_n[s] = exp(zeta_n[s]) / denom;
      // if(n == 671){
      //   Rcpp::Rcout << "s: " << s << std::endl;
      //   Rcpp::Rcout << "zeta_n[s]: " << zeta_n[s] << std::endl;
      // }
    }
    
    out_list[n] = zeta_n;
  }
  
  return(out_list);
}

//[[Rcpp::export]]
Rcpp::List theta_mean_upd_cpp(Rcpp::List theta_mean_org,
                              Rcpp::List zeta,
                              Rcpp::List samples,
                              arma::vec s_len_vec,
                              double sample_ratio) {
  
  arma::vec x_index = samples["x_index"];
  int t_len = theta_mean_org.size();
  int n_len = x_index.n_elem;
  Rcpp::List out_list(t_len);
  int s_len, t_center;
  
  for(int t = (t_len-1); t >=0; t--){
    s_len = s_len_vec[t];
    arma::vec theta_vec((s_len + 1));
    theta_vec.zeros();
    out_list[t] = theta_vec;
  }
  
  
  // arma::vec out_vec;
  for(int n = (n_len-1); n >=0; n--){
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    t_center = int(x_index[n]-1);
    
    // if(n == 0){
    //   Rcpp::Rcout << "t:" << t << std::endl;
    // }
    
    arma::vec out_vec = out_list[t_center];
    // arma::vec out_vec = out_list[t];
    
    // if(n == 0){
    //   Rcpp::Rcout << "out_vec:" << out_vec << std::endl;
    // }
    
    //
    // if(n == 0){
    //   Rcpp::Rcout << "out_vec:" << out_vec << std::endl;
    // }
    
    for(int s = (s_len-1); s >=0; s--){
      if(zeta_n[s] > 0){
        out_vec[s] += (zeta_n[s] * sample_ratio);
      }
    }
    
    out_list[t_center] = out_vec;
  }
  
  for(int t = (t_len-1); t >=0; t--){
    arma::vec theta_vec = out_list[t];
    if(arma::accu(theta_vec) == 0){
      arma::vec theta_vec_org = theta_mean_org[t];
      out_list[t] = theta_vec_org;
    }
  }
  
  return(out_list);
}

//[[Rcpp::export]]
Rcpp::List theta_var_upd_cpp(Rcpp::List theta_var_org,
                             Rcpp::List zeta,
                             arma::vec x_index,
                             arma::vec s_len_vec,
                             double sample_ratio) {
  
  int t_len = theta_var_org.size();
  int n_len = x_index.n_elem;
  Rcpp::List out_list(t_len);
  int s_len, t_center;
  
  for(int t = (t_len-1); t >=0; t--){
    s_len = s_len_vec[t];
    arma::vec theta_vec((s_len + 1));
    out_list[t] = theta_vec;
  }
  
  // Rcpp::Rcout << "--" << std::endl;
  
  // arma::vec out_vec;
  for(int n = (n_len-1); n >=0; n--){
    arma::vec zeta_n = zeta[n];
    s_len = zeta_n.n_elem;
    t_center = int(x_index[n]-1);
    
    // if(n == 0){
    //   Rcpp::Rcout << "t:" << t << std::endl;
    // }
    
    // arma::vec out_vec = out_list[t];
    arma::vec out_vec = out_list[t_center];
    
    // if(n == 0){
    //   Rcpp::Rcout << "out_vec:" << out_vec << std::endl;
    // }
    
    // 
    // if(n == 0){
    //   Rcpp::Rcout << "out_vec:" << out_vec << std::endl;
    // }
    
    for(int s = (s_len-1); s >=0; s--){
      if(zeta_n[s] > 0){
        out_vec[s] += (zeta_n[s] * (1 - zeta_n[s]) * sample_ratio);
      }
    }
    
    out_list[t_center] = out_vec;
  }
  
  for(int t = (t_len-1); t >=0; t--){
    arma::vec theta_vec = out_list[t];
    if(arma::accu(theta_vec) == 0){
      arma::vec theta_vec_org = theta_var_org[t];
      out_list[t] = theta_vec_org;
    }
  }
  
  return(out_list);
}


//[[Rcpp::export]]
Rcpp::List prune_embeddings(Rcpp::List theta_mean,
                            Rcpp::List phi_map,
                            arma::vec s_len_vec,
                            arma::mat phi,
                            arma::mat m_phi,
                            arma::mat m_hat_phi,
                            arma::mat v_phi,
                            arma::mat v_hat_phi){
  
  arma::uvec to_elem;
  Rcpp::List theta_mean_out = clone(theta_mean);
  Rcpp::List phi_map_out = clone(phi_map);
  arma::vec s_len_vec_out = s_len_vec;
  
  int t_len = theta_mean_out.size();
  int s_len;
  for(int t = (t_len-1); t >=0; t--){
    arma::uvec to_elem_t;
    s_len = s_len_vec[t];
    arma::vec theta_mean_vec = theta_mean_out[t];
    arma::vec phi_map_t = phi_map_out[t];
    for(int s = (s_len-1); s >=0; s--){
      if(theta_mean_vec[s] < 1){
        arma::uvec av(1);
        av.at(0) = phi_map_t[s] - 1;
        to_elem.insert_rows(to_elem.n_rows, av.row(0));
        
        arma::uvec av_t(1);
        av_t.at(0) = s;
        to_elem_t.insert_rows(to_elem_t.n_rows, av_t.row(0));
      }
    }
    if(to_elem_t.n_elem > 0){
      // update theta
      arma::vec theta_mean_t = theta_mean_out[t];
      theta_mean_t.shed_rows(to_elem_t);
      theta_mean_out[t] = theta_mean_t;
      
      // update s_len_vec
      s_len_vec_out[t] = s_len - to_elem_t.n_elem;
      
      // update phi_map
      arma::vec phi_map_t = phi_map_out[t];
      phi_map_t.shed_rows(to_elem_t);
      phi_map_out[t] = phi_map_t;
    }
  }
  
  phi.shed_cols(to_elem);
  m_phi.shed_cols(to_elem);
  m_hat_phi.shed_cols(to_elem);
  v_phi.shed_cols(to_elem);
  v_hat_phi.shed_cols(to_elem);
  
  // calibrate phi_map
  int s_id;
  if(to_elem.n_elem > 0){
    for(int t = (t_len-1); t >=0; t--){
      arma::vec phi_map_t = phi_map_out[t];
      s_len = phi_map_t.n_elem;
      for(int s = (s_len-1); s >=0; s--){
        s_id = phi_map_t[s];
        // if(t == 9){
        //   Rcpp::Rcout << "phi_map_t " << phi_map_t << std::endl;
        // }
        s_id = s_id - arma::accu(to_elem < s_id);
        phi_map_t[s] = s_id;
      }
      phi_map_out[t] = phi_map_t;
    }
  }
  
  Rcpp::List out = List::create(Named("theta_mean") = theta_mean_out,
                                Named("s_len_vec") = s_len_vec_out,
                                Named("to_elem") = to_elem,
                                Named("phi") = phi,
                                Named("phi_map") = phi_map_out,
                                Named("m_phi") = m_phi,
                                Named("m_hat_phi") = m_hat_phi,
                                Named("v_phi") = v_phi,
                                Named("v_hat_phi") = v_hat_phi);
  
  return(out);
}



//[[Rcpp::export]]
Rcpp::List np_norm_up_model_train_cpp(Rcpp::List np_model_init, double sigma = 1, int init_iter = 500,
                                      int epoch = 5, int n_minibatch = 100, double alpha_learn = 0.01,
                                      double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double init_sd = 0.2, 
                                      int track_data_len = 10000, double sigma_prior = 10){
  // optimization parameters
  Rcpp::List np_model = clone(np_model_init);
  Rcpp::List opt_params = np_model["opt_params"];
  opt_params = clone(opt_params);
  int iter_t = opt_params["iter_t"];
  arma::mat iter_t_phi = opt_params["iter_t_phi"];
  arma::mat m_phi = opt_params["m_phi"];
  arma::mat v_phi = opt_params["v_phi"];
  arma::mat m_hat_phi = opt_params["m_hat_phi"];
  arma::mat v_hat_phi = opt_params["m_hat_phi"];
  arma::mat m_alpha = opt_params["m_alpha"];
  arma::mat v_alpha = opt_params["v_alpha"];
  arma::mat m_hat_alpha = opt_params["m_hat_alpha"];
  arma::mat v_hat_alpha = opt_params["v_hat_alpha"];
  
  // data
  Rcpp::List data = np_model["data"];
  data = clone(data);
  int n_len_pos = data["n_len_pos"];
  int num_negative_samples = data["num_negative_samples"];
  int t_len = data["t_len"];
  int max_s_ind = t_len;
  
  Rcpp::Rcout << "sampling tracking data... " << std::endl;
  arma::uvec range = arma::linspace<arma::uvec>(0L, (n_len_pos-1), n_len_pos);
  arma::uvec traking_idx = Rcpp::RcppArmadillo::sample(range, track_data_len, false);
  Rcpp::List track_data = balanced_sampling_ccp(traking_idx, data);
  double tracking_lkhd_val;
  
  // variational parameters
  Rcpp::List var_params = np_model["var_params"];
  var_params = clone(var_params);
  
  Rcpp::List theta_mean = var_params["theta_mean"];
  theta_mean = clone(theta_mean);
  
  Rcpp::List phi_map = var_params["phi_map"];
  phi_map = clone(phi_map);
  
  arma::mat phi = var_params["phi"];
  arma::mat alpha = var_params["alpha"];
  arma::vec s_len_vec = var_params["s_len_vec"];
  
  // hyper parameters
  Rcpp::List hy_par = np_model["hy_par"];
  hy_par = clone(hy_par);
  arma::vec gamma_vec = hy_par["gamma_vec"];
  int p_len = hy_par["p_len"];
  
  int max_it = epoch * n_minibatch;
  arma::vec par_idx, elem_phi;
  arma::uvec indices_m_pos;
  Rcpp::List samples;
  int sample_len, s_len, insert_s;
  Rcpp::List zeta_pre, zeta_pos, zeta;
  Rcpp::List grad;
  Rcpp::List theta_mean_org;
  double sample_ratio;
  for(int epoch_iter = 0; epoch_iter < epoch; ++epoch_iter){
    Rcpp::Rcout << "epoch " << (epoch_iter+1) << " starts..." << std::endl;
    par_idx = Rcpp::RcppArmadillo::sample(arma::linspace(1, (n_minibatch), (n_minibatch)), n_len_pos, true);
    for(int minbatch_iter = 0; minbatch_iter < n_minibatch; ++minbatch_iter){
      // sampling
      indices_m_pos = find(par_idx == (minbatch_iter + 1));
      samples = balanced_sampling_ccp(indices_m_pos, data);
      sample_len = indices_m_pos.n_elem * (1 + num_negative_samples);
      zeta = zeta_init(sample_len);
      sample_ratio = (n_len_pos*1.0)/(indices_m_pos.n_rows * 1.0);
      // Rcpp::Rcout << "n_len_pos: "  << n_len_pos << " "<<  std::endl;
      // Rcpp::Rcout << "indices_m_pos.n_rows: "  << indices_m_pos.n_rows << " "<<  std::endl;
      // Rcpp::Rcout << "sample_ratio: "  << sample_ratio << " "<<  std::endl;
      
      if(iter_t > init_iter){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "zeta = zeta_upd_vec_up_norm_cpp(" << " "<<  std::endl;
        // }
        zeta_pre = zeta_upd_vec_up_norm_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            gamma_vec,
                                            sigma);
        
        theta_mean_org = clone(theta_mean);
        theta_mean = theta_mean_upd_cpp(theta_mean_org,
                                        zeta_pre,
                                        samples,
                                        s_len_vec,
                                        sample_ratio);
        
        // eliminate embeddings
        for(int t = (t_len-1); t >=0; t--){
          arma::uvec to_elem_t;
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          arma::vec phi_map_t = phi_map[t];
          for(int s = (s_len-1); s >=0; s--){
            if(theta_mean_vec[s] < 1){
              arma::vec av(1);
              av.at(0) = phi_map_t[s];
              elem_phi = join_cols(elem_phi, av);
              
              arma::uvec av_t(1);
              av_t.at(0) = s;
              to_elem_t.insert_rows(to_elem_t.n_rows, av_t.row(0));
            }
          }
          if(to_elem_t.n_elem > 0){
            // update theta
            arma::vec theta_mean_t = theta_mean[t];
            theta_mean_t.shed_rows(to_elem_t);
            theta_mean[t] = theta_mean_t;
            
            // update s_len_vec
            s_len_vec[t] = s_len - to_elem_t.n_elem;
            
            // update phi_map
            arma::vec phi_map_t = phi_map[t];
            phi_map_t.shed_rows(to_elem_t);
            phi_map[t] = phi_map_t;
          }
        }
        
        // add embeddings
        arma::vec zero_vec(p_len);
        zero_vec.zeros();
        
        for(int t = (t_len-1); t >=0; t--){
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          
          if(theta_mean_vec[s_len] > 1){
            // update theta
            arma::vec av(1);
            av[0] = 0;
            theta_mean_vec = join_cols(theta_mean_vec, av);
            theta_mean[t] = theta_mean_vec;
            
            // update s_len_vec
            s_len_vec[t] = s_len + 1;
            
            // update opt params
            arma::vec phi_vec = arma::randn(p_len, 1) * init_sd;
            
            if(elem_phi.n_elem > 0){
              insert_s = elem_phi[0];
              elem_phi.shed_row(0);
              
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = insert_s;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              
              phi.col((insert_s-1)) = phi_vec;
              m_phi.col((insert_s-1)) = zero_vec;
              m_hat_phi.col((insert_s-1)) = zero_vec;
              v_phi.col((insert_s-1)) = zero_vec;
              v_hat_phi.col((insert_s-1)) = zero_vec;
            }else{
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = max_s_ind + 1;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              max_s_ind = max_s_ind + 1;
              
              // phi
              phi = join_horiz(phi, phi_vec);
              
              // opt parameters
              m_phi = join_horiz(m_phi, zero_vec);
              m_hat_phi = join_horiz(m_hat_phi, zero_vec);
              v_phi = join_horiz(v_phi, zero_vec);
              v_hat_phi = join_horiz(v_hat_phi, zero_vec);
            }
          }
        }
        
        zeta_pos = zeta_upd_vec_up_norm_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            gamma_vec,
                                            sigma);
        
        zeta = zeta_normalize(zeta_pos);
        
        // Rcpp::Rcout << "phi.n_cols:" << phi.n_cols << std::endl;
      }
      
      // gradient computation
      // if(epoch_iter == 5 & minbatch_iter == 22){
      //   Rcpp::Rcout << "grad = grad_emb_norm_stoch_cpp(" << " "<<  std::endl;
      // }
      // free(grad);
      grad = grad_emb_norm_stoch_cpp(samples, phi, alpha, phi_map, zeta, sigma, t_len, sigma_prior);
      arma::mat phi_grad = grad["phi_grad"];
      arma::mat alpha_grad = grad["alpha_grad"];
      phi_grad = -1 * phi_grad;
      alpha_grad = -1 * alpha_grad;
      
      // update embeddings
      // update phi
      m_phi = beta_1 * m_phi + (1-beta_1) * phi_grad;
      v_phi = beta_2 * v_phi + (1-beta_2) * (phi_grad % phi_grad);
      m_hat_phi = m_phi / (1 - pow(beta_1, iter_t));
      v_hat_phi = v_phi / (1 - pow(beta_2, iter_t));
      phi = phi - alpha_learn * m_hat_phi / (sqrt(v_hat_phi) + epsilon);
      
      // if(iter_t == 31){
      //   Rcpp::List out = List::create(Named("s_len_vec") = s_len_vec,
      //                                 Named("theta_mean") = theta_mean,
      //                                 Named("phi") = phi);
      //   
      //   return(out);
      // }
      
      // update alpha
      m_alpha = beta_1 * m_alpha + (1-beta_1) * alpha_grad;
      v_alpha = beta_2 * v_alpha + (1-beta_2) * (alpha_grad % alpha_grad);
      m_hat_alpha = m_alpha / (1 - pow(beta_1, iter_t));
      v_hat_alpha = v_alpha / (1 - pow(beta_2, iter_t));
      alpha = alpha - alpha_learn * m_hat_alpha / (sqrt(v_hat_alpha) + epsilon);
      
      // print info to console
      Rcpp::Rcout << (minbatch_iter+1) << "/" << n_minibatch << " minibatches of epoch " << (epoch_iter+1) << " completed..." << std::endl;
      if(minbatch_iter % 10 == 9){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "tracking_lkhd_val = tracking_lkhd_norm_cpp(" << " "<<  std::endl;
        // }
        tracking_lkhd_val = tracking_lkhd_norm_cpp(track_data, phi, alpha, phi_map, s_len_vec, sigma);
        
        if(iter_t > init_iter){
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << "; Number of embeddings: " << (phi.n_cols - elem_phi.n_elem) << " "<<  std::endl;
        }else{
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << " "<<  std::endl;
        }
      }
      
      iter_t = iter_t + 1;
      iter_t_phi = iter_t_phi + 1;
    }
  }
  
  Rcpp::List tmp_out = prune_embeddings(theta_mean, phi_map, s_len_vec, phi, 
                                        m_phi, m_hat_phi, v_phi, v_hat_phi);
  
  var_params["theta_mean"] = theta_mean;
  var_params["phi_map"] = phi_map;
  var_params["s_len_vec"] = s_len_vec;
  var_params["phi"] = phi;
  var_params["alpha"] = alpha;
  var_params["zeta"] = zeta;
  
  Rcpp::List out = List::create(Named("var_params") = var_params);
  
  return(out);
}


//[[Rcpp::export]]
Rcpp::List np_pois_up_model_train_cpp(Rcpp::List np_model_init, int init_iter = 500,
                                      int epoch = 5, int n_minibatch = 100, double alpha_learn = 0.01,
                                      double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double init_sd = 0.2, 
                                      int track_data_len = 10000, double sigma_prior = 10){
  // optimization parameters
  Rcpp::List np_model = clone(np_model_init);
  Rcpp::List opt_params = np_model["opt_params"];
  opt_params = clone(opt_params);
  int iter_t = opt_params["iter_t"];
  arma::mat iter_t_phi = opt_params["iter_t_phi"];
  arma::mat m_phi = opt_params["m_phi"];
  arma::mat v_phi = opt_params["v_phi"];
  arma::mat m_hat_phi = opt_params["m_hat_phi"];
  arma::mat v_hat_phi = opt_params["m_hat_phi"];
  arma::mat m_alpha = opt_params["m_alpha"];
  arma::mat v_alpha = opt_params["v_alpha"];
  arma::mat m_hat_alpha = opt_params["m_hat_alpha"];
  arma::mat v_hat_alpha = opt_params["v_hat_alpha"];
  
  // data
  Rcpp::List data = np_model["data"];
  data = clone(data);
  int n_len_pos = data["n_len_pos"];
  int num_negative_samples = data["num_negative_samples"];
  int t_len = data["t_len"];
  int max_s_ind = t_len;
  
  Rcpp::Rcout << "sampling tracking data... " << std::endl;
  arma::uvec range = arma::linspace<arma::uvec>(0L, (n_len_pos-1), n_len_pos);
  arma::uvec traking_idx = Rcpp::RcppArmadillo::sample(range, track_data_len, false);
  Rcpp::List track_data = balanced_sampling_ccp(traking_idx, data);
  double tracking_lkhd_val;
  
  // variational parameters
  Rcpp::List var_params = np_model["var_params"];
  var_params = clone(var_params);
  
  Rcpp::List theta_mean = var_params["theta_mean"];
  theta_mean = clone(theta_mean);
  
  Rcpp::List phi_map = var_params["phi_map"];
  phi_map = clone(phi_map);
  
  arma::mat phi = var_params["phi"];
  arma::mat alpha = var_params["alpha"];
  arma::vec s_len_vec = var_params["s_len_vec"];
  
  // hyper parameters
  Rcpp::List hy_par = np_model["hy_par"];
  hy_par = clone(hy_par);
  arma::vec gamma_vec = hy_par["gamma_vec"];
  int p_len = hy_par["p_len"];
  
  int max_it = epoch * n_minibatch;
  arma::vec par_idx, elem_phi;
  arma::uvec indices_m_pos;
  Rcpp::List samples;
  int sample_len, s_len, insert_s;
  Rcpp::List zeta_pre, zeta_pos, zeta;
  Rcpp::List grad;
  Rcpp::List theta_mean_org;
  double sample_ratio;
  for(int epoch_iter = 0; epoch_iter < epoch; ++epoch_iter){
    Rcpp::Rcout << "epoch " << (epoch_iter+1) << " starts..." << std::endl;
    par_idx = Rcpp::RcppArmadillo::sample(arma::linspace(1, (n_minibatch), (n_minibatch)), n_len_pos, true);
    for(int minbatch_iter = 0; minbatch_iter < n_minibatch; ++minbatch_iter){
      // sampling
      indices_m_pos = find(par_idx == (minbatch_iter + 1));
      samples = balanced_sampling_ccp(indices_m_pos, data);
      sample_len = indices_m_pos.n_elem * (1 + num_negative_samples);
      zeta = zeta_init(sample_len);
      sample_ratio = (n_len_pos*1.0)/(indices_m_pos.n_rows * 1.0);
      // Rcpp::Rcout << "n_len_pos: "  << n_len_pos << " "<<  std::endl;
      // Rcpp::Rcout << "indices_m_pos.n_rows: "  << indices_m_pos.n_rows << " "<<  std::endl;
      // Rcpp::Rcout << "sample_ratio: "  << sample_ratio << " "<<  std::endl;
      
      if(iter_t > init_iter){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "zeta = zeta_upd_vec_up_norm_cpp(" << " "<<  std::endl;
        // }
        zeta_pre = zeta_upd_vec_up_pois_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            gamma_vec);
        
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "theta_mean = theta_mean_upd_cpp(" << " "<<  std::endl;
        //   
        //   Rcpp::List out = List::create(Named("phi_map") = phi_map,
        //                                 Named("theta_mean") = theta_mean,
        //                                 Named("zeta") = zeta,
        //                                 Named("samples") = samples,
        //                                 Named("s_len_vec") = s_len_vec,
        //                                 Named("sample_ratio") = sample_ratio);
        //   
        //   return(out);
        // }
        
        theta_mean_org = clone(theta_mean);
        theta_mean = theta_mean_upd_cpp(theta_mean_org,
                                        zeta_pre,
                                        samples,
                                        s_len_vec,
                                        sample_ratio);
        
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "tmp_out = prune_embeddings(" << " "<<  std::endl;
        // }
        // tmp_prune = prune_embeddings(theta_mean_upd, phi_map, s_len_vec, phi, 
        //                              m_phi, m_hat_phi, v_phi, v_hat_phi);
        // Rcpp::List theta_mean_pruned = tmp_prune["theta_mean"];
        // theta_mean_pruned = clone(theta_mean_pruned);
        // Rcpp::List phi_map_pruned = tmp_prune["phi_map"];
        // phi_map_pruned = clone(phi_map_pruned);
        // arma::vec s_len_vec_pruned = tmp_prune["s_len_vec"];
        // arma::mat phi_pruned = tmp_prune["phi"];
        // arma::mat m_phi_pruned = tmp_prune["m_phi"];
        // arma::mat m_hat_phi_pruned = tmp_prune["m_hat_phi"];
        // arma::mat v_phi_pruned = tmp_prune["v_phi"];
        // arma::mat v_hat_phi_pruned = tmp_prune["v_hat_phi"];
        
        // prune embeddings
        for(int t = (t_len-1); t >=0; t--){
          arma::uvec to_elem_t;
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          arma::vec phi_map_t = phi_map[t];
          for(int s = (s_len-1); s >=0; s--){
            if(theta_mean_vec[s] < 1){
              arma::vec av(1);
              av.at(0) = phi_map_t[s];
              elem_phi = join_cols(elem_phi, av);
              
              arma::uvec av_t(1);
              av_t.at(0) = s;
              to_elem_t.insert_rows(to_elem_t.n_rows, av_t.row(0));
            }
          }
          if(to_elem_t.n_elem > 0){
            // update theta
            arma::vec theta_mean_t = theta_mean[t];
            theta_mean_t.shed_rows(to_elem_t);
            theta_mean[t] = theta_mean_t;
            
            // update s_len_vec
            s_len_vec[t] = s_len - to_elem_t.n_elem;
            
            // update phi_map
            arma::vec phi_map_t = phi_map[t];
            phi_map_t.shed_rows(to_elem_t);
            phi_map[t] = phi_map_t;
          }
        }
        
        // add embeddings
        arma::vec zero_vec(p_len);
        zero_vec.zeros();
        
        for(int t = (t_len-1); t >=0; t--){
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          
          if(theta_mean_vec[s_len] > 1){
            // update theta
            arma::vec av(1);
            av[0] = 0;
            theta_mean_vec = join_cols(theta_mean_vec, av);
            theta_mean[t] = theta_mean_vec;
            
            if(elem_phi.n_elem > 0){
              insert_s = elem_phi[0];
              elem_phi.shed_row(0);
              
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = insert_s;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              
              // phi
              arma::vec phi_vec = arma::randn(p_len, 1) * init_sd;
              phi.col((insert_s-1)) = phi_vec;
              
              // opt params
              m_phi.col((insert_s-1)) = zero_vec;
              m_hat_phi.col((insert_s-1)) = zero_vec;
              v_phi.col((insert_s-1)) = zero_vec;
              v_hat_phi.col((insert_s-1)) = zero_vec;
              
              // update s_len_vec
              s_len_vec[t] = s_len + 1;
            }else{
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = max_s_ind + 1;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              
              // phi
              arma::vec phi_vec = arma::randn(p_len, 1) * init_sd;
              phi = join_horiz(phi, phi_vec);
              
              // opt parameters
              m_phi = join_horiz(m_phi, zero_vec);
              m_hat_phi = join_horiz(m_hat_phi, zero_vec);
              v_phi = join_horiz(v_phi, zero_vec);
              v_hat_phi = join_horiz(v_hat_phi, zero_vec);
              
              // update s_len_vec
              s_len_vec[t] = s_len + 1;
              max_s_ind = max_s_ind + 1;
            }
          }
        }
        
        zeta_pos = zeta_upd_vec_up_pois_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            gamma_vec);
        
        zeta = zeta_normalize(zeta_pos);
        
        // Rcpp::Rcout << "phi.n_cols:" << phi.n_cols << std::endl;
      }
      
      // gradient computation
      // if(epoch_iter == 5 & minbatch_iter == 22){
      //   Rcpp::Rcout << "grad = grad_emb_norm_stoch_cpp(" << " "<<  std::endl;
      // }
      // free(grad);
      grad = grad_emb_pois_stoch_cpp(samples, phi, alpha, phi_map, zeta, t_len, sigma_prior);
      arma::mat phi_grad = grad["phi_grad"];
      arma::mat alpha_grad = grad["alpha_grad"];
      phi_grad = -1 * phi_grad;
      alpha_grad = -1 * alpha_grad;
      
      // update embeddings
      // update phi
      m_phi = beta_1 * m_phi + (1-beta_1) * phi_grad;
      v_phi = beta_2 * v_phi + (1-beta_2) * (phi_grad % phi_grad);
      m_hat_phi = m_phi / (1 - pow(beta_1, iter_t));
      v_hat_phi = v_phi / (1 - pow(beta_2, iter_t));
      phi = phi - alpha_learn * m_hat_phi / (sqrt(v_hat_phi) + epsilon);
      
      // if(iter_t == 31){
      //   Rcpp::List out = List::create(Named("s_len_vec") = s_len_vec,
      //                                 Named("theta_mean") = theta_mean,
      //                                 Named("phi") = phi);
      //   
      //   return(out);
      // }
      
      // update alpha
      m_alpha = beta_1 * m_alpha + (1-beta_1) * alpha_grad;
      v_alpha = beta_2 * v_alpha + (1-beta_2) * (alpha_grad % alpha_grad);
      m_hat_alpha = m_alpha / (1 - pow(beta_1, iter_t));
      v_hat_alpha = v_alpha / (1 - pow(beta_2, iter_t));
      alpha = alpha - alpha_learn * m_hat_alpha / (sqrt(v_hat_alpha) + epsilon);
      
      // print info to console
      Rcpp::Rcout << (minbatch_iter+1) << "/" << n_minibatch << " minibatches of epoch " << (epoch_iter+1) << " completed..." << std::endl;
      if(minbatch_iter % 10 == 9){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "tracking_lkhd_val = tracking_lkhd_norm_cpp(" << " "<<  std::endl;
        // }
        tracking_lkhd_val = tracking_lkhd_pois_cpp(track_data, phi, alpha, phi_map, s_len_vec);
        
        if(iter_t > init_iter){
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << "; Number of embeddings: " << (phi.n_cols - elem_phi.n_elem) << " "<<  std::endl;
        }else{
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << " "<<  std::endl;
        }
      }
      
      iter_t = iter_t + 1;
      iter_t_phi = iter_t_phi + 1;
    }
  }
  
  // final output
  var_params["theta_mean"] = theta_mean;
  var_params["phi_map"] = phi_map;
  var_params["s_len_vec"] = s_len_vec;
  var_params["phi"] = phi;
  var_params["alpha"] = alpha;
  var_params["zeta"] = zeta;
  
  Rcpp::List out = List::create(Named("var_params") = var_params);
  
  return(out);
}



//[[Rcpp::export]]
Rcpp::List np_norm_dp_model_train_cpp(Rcpp::List np_model_init, double sigma = 1, int init_iter = 500,
                                      int epoch = 5, int n_minibatch = 100, double alpha_learn = 0.01,
                                      double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double init_sd = 0.2, 
                                      int track_data_len = 10000, double sigma_prior = 10){
  // optimization parameters
  Rcpp::List np_model = clone(np_model_init);
  Rcpp::List opt_params = np_model["opt_params"];
  opt_params = clone(opt_params);
  int iter_t = opt_params["iter_t"];
  arma::mat iter_t_phi = opt_params["iter_t_phi"];
  arma::mat m_phi = opt_params["m_phi"];
  arma::mat v_phi = opt_params["v_phi"];
  arma::mat m_hat_phi = opt_params["m_hat_phi"];
  arma::mat v_hat_phi = opt_params["m_hat_phi"];
  arma::mat m_alpha = opt_params["m_alpha"];
  arma::mat v_alpha = opt_params["v_alpha"];
  arma::mat m_hat_alpha = opt_params["m_hat_alpha"];
  arma::mat v_hat_alpha = opt_params["v_hat_alpha"];
  
  // data
  Rcpp::List data = np_model["data"];
  data = clone(data);
  arma::vec item_cnt = data["item_cnt"];
  int n_len_pos = data["n_len_pos"];
  int num_negative_samples = data["num_negative_samples"];
  int t_len = data["t_len"];
  int max_s_ind = t_len;
  
  Rcpp::Rcout << "sampling tracking data... " << std::endl;
  arma::uvec range = arma::linspace<arma::uvec>(0L, (n_len_pos-1), n_len_pos);
  arma::uvec traking_idx = Rcpp::RcppArmadillo::sample(range, track_data_len, false);
  Rcpp::List track_data = balanced_sampling_ccp(traking_idx, data);
  double tracking_lkhd_val;
  
  // variational parameters
  Rcpp::List var_params = np_model["var_params"];
  var_params = clone(var_params);
  
  Rcpp::List theta_mean = var_params["theta_mean"];
  theta_mean = clone(theta_mean);
  
  Rcpp::List theta_var = var_params["theta_var"];
  theta_var = clone(theta_var);
  
  Rcpp::List phi_map = var_params["phi_map"];
  phi_map = clone(phi_map);
  
  arma::mat phi = var_params["phi"];
  arma::mat alpha = var_params["alpha"];
  arma::vec s_len_vec = var_params["s_len_vec"];
  
  // hyper parameters
  Rcpp::List hy_par = np_model["hy_par"];
  hy_par = clone(hy_par);
  arma::vec gamma_vec = hy_par["gamma_vec"];
  int p_len = hy_par["p_len"];
  
  int max_it = epoch * n_minibatch;
  arma::vec par_idx, elem_phi;
  arma::uvec indices_m_pos;
  Rcpp::List samples;
  int sample_len, s_len, insert_s;
  Rcpp::List zeta_pre, zeta_pos, zeta;
  Rcpp::List grad;
  Rcpp::List theta_mean_org, theta_var_org;
  double sample_ratio;
  for(int epoch_iter = 0; epoch_iter < epoch; ++epoch_iter){
    Rcpp::Rcout << "epoch " << (epoch_iter+1) << " starts..." << std::endl;
    par_idx = Rcpp::RcppArmadillo::sample(arma::linspace(1, (n_minibatch), (n_minibatch)), n_len_pos, true);
    for(int minbatch_iter = 0; minbatch_iter < n_minibatch; ++minbatch_iter){
      // sampling
      indices_m_pos = find(par_idx == (minbatch_iter + 1));
      samples = balanced_sampling_ccp(indices_m_pos, data);
      sample_len = indices_m_pos.n_elem * (1 + num_negative_samples);
      zeta = zeta_init(sample_len);
      sample_ratio = (n_len_pos*1.0)/(indices_m_pos.n_rows * 1.0);
      // Rcpp::Rcout << "n_len_pos: "  << n_len_pos << " "<<  std::endl;
      // Rcpp::Rcout << "indices_m_pos.n_rows: "  << indices_m_pos.n_rows << " "<<  std::endl;
      // Rcpp::Rcout << "sample_ratio: "  << sample_ratio << " "<<  std::endl;
      
      if(iter_t > init_iter){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "zeta = zeta_upd_vec_up_norm_cpp(" << " "<<  std::endl;
        // }
        zeta_pre = zeta_upd_vec_dp_norm_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            theta_mean,
                                            theta_var,
                                            item_cnt,
                                            gamma_vec,
                                            sigma);
        
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "theta_mean = theta_mean_upd_cpp(" << " "<<  std::endl;
        //   
        //   Rcpp::List out = List::create(Named("phi_map") = phi_map,
        //                                 Named("theta_mean") = theta_mean,
        //                                 Named("zeta") = zeta,
        //                                 Named("samples") = samples,
        //                                 Named("s_len_vec") = s_len_vec,
        //                                 Named("sample_ratio") = sample_ratio);
        //   
        //   return(out);
        // }
        
        theta_mean_org = clone(theta_mean);
        theta_mean = theta_mean_upd_cpp(theta_mean_org,
                                        zeta_pre,
                                        samples,
                                        s_len_vec,
                                        sample_ratio);
        
        
        theta_var_org = clone(theta_var);
        theta_var = theta_mean_upd_cpp(theta_var_org,
                                       zeta_pre,
                                       samples,
                                       s_len_vec,
                                       sample_ratio);
        
        // prune embeddings
        // if(iter_t == 109){
        //   Rcpp::Rcout << "prune embeddings" << std::endl;
        // }
        
        for(int t = (t_len-1); t >=0; t--){
          arma::uvec to_elem_t;
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          arma::vec phi_map_t = phi_map[t];
          for(int s = (s_len-1); s >=0; s--){
            if(theta_mean_vec[s] < 1){
              arma::vec av(1);
              av.at(0) = phi_map_t[s];
              elem_phi = join_cols(elem_phi, av);
              
              arma::uvec av_t(1);
              av_t.at(0) = s;
              to_elem_t.insert_rows(to_elem_t.n_rows, av_t.row(0));
            }
          }
          if(to_elem_t.n_elem > 0){
            // update theta
            arma::vec theta_mean_t = theta_mean[t];
            theta_mean_t.shed_rows(to_elem_t);
            theta_mean[t] = theta_mean_t;
            
            // update s_len_vec
            s_len_vec[t] = s_len - to_elem_t.n_elem;
            
            // update phi_map
            arma::vec phi_map_t = phi_map[t];
            phi_map_t.shed_rows(to_elem_t);
            phi_map[t] = phi_map_t;
          }
        }
        
        // add embeddings
        // if(iter_t == 109){
        //   Rcpp::Rcout << "prune embeddings" << std::endl;
        // }
        arma::vec zero_vec(p_len);
        zero_vec.zeros();
        
        for(int t = (t_len-1); t >=0; t--){
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          
          // if(iter_t == 109 & t == 2545){
          //   Rcpp::Rcout << "s_len: " << s_len  << std::endl;
          //   Rcpp::Rcout << "theta_mean_vec: " << theta_mean_vec.t()  << std::endl;
          // }
          
          if(theta_mean_vec[s_len] > 1){
            // update theta
            // if(iter_t == 109 & t == 2545){
            //   Rcpp::Rcout << "update theta" << std::endl;
            // }
            
            arma::vec av(1);
            av[0] = 0;
            theta_mean_vec = join_cols(theta_mean_vec, av);
            theta_mean[t] = theta_mean_vec;
            
            // update s_len_vec
            // if(iter_t == 109 & t == 2545){
            //   Rcpp::Rcout << "update s_len_vec" << std::endl;
            // }
            s_len_vec[t] = s_len + 1;
            
            // update opt params
            arma::vec phi_vec = arma::randn(p_len, 1) * init_sd;
            
            if(elem_phi.n_elem > 0){
              insert_s = elem_phi[0];
              elem_phi.shed_row(0);
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "insert_s" << insert_s << std::endl;
              // }
              
              // phi_map
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "phi_map (insert)" << std::endl;
              // }
              arma::vec phi_map_t = phi_map[t];
              av[0] = insert_s;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "phi (insert)" << std::endl;
              // }
              phi.col((insert_s-1)) = phi_vec;
              
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "m_phi (insert)" << std::endl;
              // }
              m_phi.col((insert_s-1)) = zero_vec;
              
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "m_hat_phi (insert)" << std::endl;
              // }
              m_hat_phi.col((insert_s-1)) = zero_vec;
              
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "v_phi (insert)" << std::endl;
              // }
              v_phi.col((insert_s-1)) = zero_vec;
              
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "v_hat_phi (insert)" << std::endl;
              // }
              v_hat_phi.col((insert_s-1)) = zero_vec;
            }else{
              // phi_map
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "phi_map (append)" << std::endl;
              // }
              arma::vec phi_map_t = phi_map[t];
              av[0] = max_s_ind + 1;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              max_s_ind = max_s_ind + 1;
              
              // phi
              // if(iter_t == 109 & t == 2545){
              //   Rcpp::Rcout << "phi (append)" << std::endl;
              // }
              phi = join_horiz(phi, phi_vec);
              
              // opt parameters
              m_phi = join_horiz(m_phi, zero_vec);
              m_hat_phi = join_horiz(m_hat_phi, zero_vec);
              v_phi = join_horiz(v_phi, zero_vec);
              v_hat_phi = join_horiz(v_hat_phi, zero_vec);
            }
          }
        }
        
        zeta_pos = zeta_upd_vec_dp_norm_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            theta_mean,
                                            theta_var,
                                            item_cnt,
                                            gamma_vec,
                                            sigma);
        
        zeta = zeta_normalize(zeta_pos);
        
        // Rcpp::Rcout << "phi.n_cols:" << phi.n_cols << std::endl;
      }
      
      // gradient computation
      // if(epoch_iter == 5 & minbatch_iter == 22){
      //   Rcpp::Rcout << "grad = grad_emb_norm_stoch_cpp(" << " "<<  std::endl;
      // }
      // free(grad);
      grad = grad_emb_norm_stoch_cpp(samples, phi, alpha, phi_map, zeta, sigma, t_len, sigma_prior);
      arma::mat phi_grad = grad["phi_grad"];
      arma::mat alpha_grad = grad["alpha_grad"];
      phi_grad = -1 * phi_grad;
      alpha_grad = -1 * alpha_grad;
      
      // update embeddings
      // update phi
      m_phi = beta_1 * m_phi + (1-beta_1) * phi_grad;
      v_phi = beta_2 * v_phi + (1-beta_2) * (phi_grad % phi_grad);
      m_hat_phi = m_phi / (1 - pow(beta_1, iter_t));
      v_hat_phi = v_phi / (1 - pow(beta_2, iter_t));
      phi = phi - alpha_learn * m_hat_phi / (sqrt(v_hat_phi) + epsilon);
      
      // if(iter_t == 31){
      //   Rcpp::List out = List::create(Named("s_len_vec") = s_len_vec,
      //                                 Named("theta_mean") = theta_mean,
      //                                 Named("phi") = phi);
      //   
      //   return(out);
      // }
      
      // update alpha
      m_alpha = beta_1 * m_alpha + (1-beta_1) * alpha_grad;
      v_alpha = beta_2 * v_alpha + (1-beta_2) * (alpha_grad % alpha_grad);
      m_hat_alpha = m_alpha / (1 - pow(beta_1, iter_t));
      v_hat_alpha = v_alpha / (1 - pow(beta_2, iter_t));
      alpha = alpha - alpha_learn * m_hat_alpha / (sqrt(v_hat_alpha) + epsilon);
      
      // print info to console
      Rcpp::Rcout << (minbatch_iter+1) << "/" << n_minibatch << " minibatches of epoch " << (epoch_iter+1) << " completed..." << std::endl;
      if(minbatch_iter % 10 == 9){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "tracking_lkhd_val = tracking_lkhd_norm_cpp(" << " "<<  std::endl;
        // }
        tracking_lkhd_val = tracking_lkhd_norm_cpp(track_data, phi, alpha, phi_map, s_len_vec, sigma);
        
        if(iter_t > init_iter){
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << "; Number of embeddings: " << (phi.n_cols - elem_phi.n_elem) << " "<<  std::endl;
        }else{
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << " "<<  std::endl;
        }
      }
      
      iter_t = iter_t + 1;
      iter_t_phi = iter_t_phi + 1;
    }
  }
  
  // final output
  var_params["theta_mean"] = theta_mean;
  var_params["phi_map"] = phi_map;
  var_params["s_len_vec"] = s_len_vec;
  var_params["phi"] = phi;
  var_params["alpha"] = alpha;
  var_params["zeta"] = zeta;
  
  Rcpp::List out = List::create(Named("var_params") = var_params);
  
  return(out);
}


//[[Rcpp::export]]
Rcpp::List np_pois_dp_model_train_cpp(Rcpp::List np_model_init, int init_iter = 500,
                                      int epoch = 5, int n_minibatch = 100, double alpha_learn = 0.01,
                                      double beta_1 = 0.9, double beta_2 = 0.999, double epsilon = 1e-8, double init_sd = 0.2, 
                                      int track_data_len = 10000, double sigma_prior = 10){
  // optimization parameters
  Rcpp::List np_model = clone(np_model_init);
  Rcpp::List opt_params = np_model["opt_params"];
  opt_params = clone(opt_params);
  int iter_t = opt_params["iter_t"];
  arma::mat iter_t_phi = opt_params["iter_t_phi"];
  arma::mat m_phi = opt_params["m_phi"];
  arma::mat v_phi = opt_params["v_phi"];
  arma::mat m_hat_phi = opt_params["m_hat_phi"];
  arma::mat v_hat_phi = opt_params["m_hat_phi"];
  arma::mat m_alpha = opt_params["m_alpha"];
  arma::mat v_alpha = opt_params["v_alpha"];
  arma::mat m_hat_alpha = opt_params["m_hat_alpha"];
  arma::mat v_hat_alpha = opt_params["v_hat_alpha"];
  
  // data
  Rcpp::List data = np_model["data"];
  data = clone(data);
  arma::vec item_cnt = data["item_cnt"];
  int n_len_pos = data["n_len_pos"];
  int num_negative_samples = data["num_negative_samples"];
  int t_len = data["t_len"];
  int max_s_ind = t_len;
  
  Rcpp::Rcout << "sampling tracking data... " << std::endl;
  arma::uvec range = arma::linspace<arma::uvec>(0L, (n_len_pos-1), n_len_pos);
  arma::uvec traking_idx = Rcpp::RcppArmadillo::sample(range, track_data_len, false);
  Rcpp::List track_data = balanced_sampling_ccp(traking_idx, data);
  double tracking_lkhd_val;
  
  // variational parameters
  Rcpp::List var_params = np_model["var_params"];
  var_params = clone(var_params);
  
  Rcpp::List theta_mean = var_params["theta_mean"];
  theta_mean = clone(theta_mean);
  
  Rcpp::List theta_var = var_params["theta_var"];
  theta_var = clone(theta_var);
  
  Rcpp::List phi_map = var_params["phi_map"];
  phi_map = clone(phi_map);
  
  arma::mat phi = var_params["phi"];
  arma::mat alpha = var_params["alpha"];
  arma::vec s_len_vec = var_params["s_len_vec"];
  
  // hyper parameters
  Rcpp::List hy_par = np_model["hy_par"];
  hy_par = clone(hy_par);
  arma::vec gamma_vec = hy_par["gamma_vec"];
  int p_len = hy_par["p_len"];
  
  int max_it = epoch * n_minibatch;
  arma::vec par_idx, elem_phi;
  arma::uvec indices_m_pos;
  Rcpp::List samples;
  int sample_len, s_len, insert_s;
  Rcpp::List zeta_pre, zeta_pos, zeta;
  Rcpp::List grad;
  Rcpp::List theta_mean_org, theta_var_org;
  double sample_ratio;
  for(int epoch_iter = 0; epoch_iter < epoch; ++epoch_iter){
    // Rcpp::Rcout << "epoch " << (epoch_iter+1) << " starts..." << std::endl;
    par_idx = Rcpp::RcppArmadillo::sample(arma::linspace(1, (n_minibatch), (n_minibatch)), n_len_pos, true);
    for(int minbatch_iter = 0; minbatch_iter < n_minibatch; ++minbatch_iter){
      // sampling
      // if(iter_t >= 100){
      //   Rcpp::Rcout << "iter_t" << iter_t << std::endl;
      // }
      // if(iter_t >= 100){
      //   Rcpp::Rcout << "sampling" << std::endl;
      // }
      indices_m_pos = find(par_idx == (minbatch_iter + 1));
      samples = balanced_sampling_ccp(indices_m_pos, data);
      sample_len = indices_m_pos.n_elem * (1 + num_negative_samples);
      zeta = zeta_init(sample_len);
      sample_ratio = (n_len_pos*1.0)/(indices_m_pos.n_rows * 1.0);
      
      if(iter_t > init_iter){
        // if(iter_t >= 100){
        //   Rcpp::Rcout << "update zeta" << std::endl;
        // }
        zeta_pre = zeta_upd_vec_dp_pois_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            theta_mean,
                                            theta_var,
                                            item_cnt,
                                            gamma_vec);
        
        
        // if(iter_t >= 100){
        //   Rcpp::Rcout << "update theta" << std::endl;
        // }
        
        theta_mean_org = clone(theta_mean);
        theta_mean = theta_mean_upd_cpp(theta_mean_org,
                                        zeta_pre,
                                        samples,
                                        s_len_vec,
                                        sample_ratio);
        
        
        theta_var_org = clone(theta_var);
        theta_var = theta_mean_upd_cpp(theta_var_org,
                                       zeta_pre,
                                       samples,
                                       s_len_vec,
                                       sample_ratio);
        
        // prune embeddings
        // eliminate embeddings
        // if(iter_t >= 109){
        //   Rcpp::Rcout << "eliminate embeddings" << std::endl;
        // }
        
        for(int t = (t_len-1); t >=0; t--){
          // if(iter_t >= 109){
          //   Rcpp::Rcout << "t" << t << std::endl;
          // }
          
          arma::uvec to_elem_t;
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          arma::vec phi_map_t = phi_map[t];
          for(int s = (s_len-1); s >=0; s--){
            if(theta_mean_vec[s] < 1){
              arma::vec av(1);
              av.at(0) = phi_map_t[s];
              elem_phi = join_cols(elem_phi, av);
              
              arma::uvec av_t(1);
              av_t.at(0) = s;
              to_elem_t.insert_rows(to_elem_t.n_rows, av_t.row(0));
            }
          }
          if(to_elem_t.n_elem > 0){
            // update theta
            arma::vec theta_mean_t = theta_mean[t];
            theta_mean_t.shed_rows(to_elem_t);
            theta_mean[t] = theta_mean_t;
            
            // update s_len_vec
            s_len_vec[t] = s_len - to_elem_t.n_elem;
            
            // update phi_map
            arma::vec phi_map_t = phi_map[t];
            phi_map_t.shed_rows(to_elem_t);
            phi_map[t] = phi_map_t;
          }
        }
        
        // add embeddings
        // if(iter_t == 109){
        //   Rcpp::Rcout << "add embeddings" << std::endl;
        // }
        
        arma::vec zero_vec(p_len);
        zero_vec.zeros();
        
        for(int t = (t_len-1); t >=0; t--){
          // if(iter_t == 109){
          //   Rcpp::Rcout << "t" << t << std::endl;
          // }
          s_len = s_len_vec[t];
          arma::vec theta_mean_vec = theta_mean[t];
          
          if(theta_mean_vec[s_len] > 1){
            // update theta
            arma::vec av(1);
            av[0] = 0;
            theta_mean_vec = join_cols(theta_mean_vec, av);
            theta_mean[t] = theta_mean_vec;
            
            // update s_len_vec
            s_len_vec[t] = s_len + 1;
            
            // update opt params
            arma::vec phi_vec = arma::randn(p_len, 1) * init_sd;
            
            if(elem_phi.n_elem > 0){
              insert_s = elem_phi[0];
              elem_phi.shed_row(0);
              
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = insert_s;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              
              phi.col((insert_s-1)) = phi_vec;
              m_phi.col((insert_s-1)) = zero_vec;
              m_hat_phi.col((insert_s-1)) = zero_vec;
              v_phi.col((insert_s-1)) = zero_vec;
              v_hat_phi.col((insert_s-1)) = zero_vec;
            }else{
              // phi_map
              arma::vec phi_map_t = phi_map[t];
              av[0] = max_s_ind + 1;
              phi_map_t = join_cols(phi_map_t, av);
              phi_map[t] = phi_map_t;
              max_s_ind = max_s_ind + 1;
              
              // phi
              phi = join_horiz(phi, phi_vec);
              
              // opt parameters
              m_phi = join_horiz(m_phi, zero_vec);
              m_hat_phi = join_horiz(m_hat_phi, zero_vec);
              v_phi = join_horiz(v_phi, zero_vec);
              v_hat_phi = join_horiz(v_hat_phi, zero_vec);
            }
          }
        }
        
        
        zeta_pos = zeta_upd_vec_dp_pois_cpp(samples,
                                            phi,
                                            alpha,
                                            phi_map,
                                            s_len_vec,
                                            theta_mean,
                                            theta_var,
                                            item_cnt,
                                            gamma_vec);
        
        zeta = zeta_normalize(zeta_pos);
        
        // Rcpp::Rcout << "phi.n_cols:" << phi.n_cols << std::endl;
      }
      
      // gradient computation
      // if(epoch_iter == 5 & minbatch_iter == 22){
      //   Rcpp::Rcout << "grad = grad_emb_norm_stoch_cpp(" << " "<<  std::endl;
      // }
      // free(grad);
      // if(iter_t == 109){
      //   Rcpp::Rcout << "compute gradient" << std::endl;
      // }
      grad = grad_emb_pois_stoch_cpp(samples, phi, alpha, phi_map, zeta, t_len, sigma_prior);
      arma::mat phi_grad = grad["phi_grad"];
      arma::mat alpha_grad = grad["alpha_grad"];
      phi_grad = -1 * phi_grad;
      alpha_grad = -1 * alpha_grad;
      
      // update embeddings
      // update phi
      m_phi = beta_1 * m_phi + (1-beta_1) * phi_grad;
      v_phi = beta_2 * v_phi + (1-beta_2) * (phi_grad % phi_grad);
      m_hat_phi = m_phi / (1 - pow(beta_1, iter_t));
      v_hat_phi = v_phi / (1 - pow(beta_2, iter_t));
      phi = phi - alpha_learn * m_hat_phi / (sqrt(v_hat_phi) + epsilon);
      
      // if(iter_t == 31){
      //   Rcpp::List out = List::create(Named("s_len_vec") = s_len_vec,
      //                                 Named("theta_mean") = theta_mean,
      //                                 Named("phi") = phi);
      //   
      //   return(out);
      // }
      
      // update alpha
      m_alpha = beta_1 * m_alpha + (1-beta_1) * alpha_grad;
      v_alpha = beta_2 * v_alpha + (1-beta_2) * (alpha_grad % alpha_grad);
      m_hat_alpha = m_alpha / (1 - pow(beta_1, iter_t));
      v_hat_alpha = v_alpha / (1 - pow(beta_2, iter_t));
      alpha = alpha - alpha_learn * m_hat_alpha / (sqrt(v_hat_alpha) + epsilon);
      
      // print info to console
      Rcpp::Rcout << (minbatch_iter+1) << "/" << n_minibatch << " minibatches of epoch " << (epoch_iter+1) << " completed..." << std::endl;
      if(minbatch_iter % 10 == 9){
        // if(epoch_iter == 5 & minbatch_iter == 22){
        //   Rcpp::Rcout << "tracking_lkhd_val = tracking_lkhd_norm_cpp(" << " "<<  std::endl;
        // }
        // if(iter_t == 109){
        //   Rcpp::Rcout << "copmute tracking likelihood" << std::endl;
        // }
        tracking_lkhd_val = tracking_lkhd_pois_cpp(track_data, phi, alpha, phi_map, s_len_vec);
        
        if(iter_t > init_iter){
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << "; Number of embeddings: " << (phi.n_cols - elem_phi.n_elem) << " "<<  std::endl;
        }else{
          Rcpp::Rcout << "log-likelihood per locus: " << (tracking_lkhd_val / track_data_len) << " "<<  std::endl;
        }
      }
      
      iter_t = iter_t + 1;
      iter_t_phi = iter_t_phi + 1;
    }
  }
  
  // final output
  var_params["theta_mean"] = theta_mean;
  var_params["phi_map"] = phi_map;
  var_params["s_len_vec"] = s_len_vec;
  var_params["phi"] = phi;
  var_params["alpha"] = alpha;
  var_params["zeta"] = zeta;
  
  Rcpp::List out = List::create(Named("var_params") = var_params);
  
  return(out);
}

// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
*/
