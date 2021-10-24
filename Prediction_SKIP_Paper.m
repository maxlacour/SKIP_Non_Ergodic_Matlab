%% Imports Data

clear all


Imports_Data_California_100000



n_repmat = 1; % This is to replicate the available dataset to check how fast the code is


Eq_ID_vector = repmat(Eq_ID_vector, [n_repmat 1]);
Region_vector = repmat(Region_vector, [n_repmat 1]);
Mag_vector = repmat(Mag_vector, [n_repmat 1]);
RJB_vector = repmat(RJB_vector, [n_repmat 1]);
VS30_vector = repmat(VS30_vector, [n_repmat 1]);
SOF_vector = repmat(SOF_vector, [n_repmat 1]);
log_PGA_vector = repmat(log_PGA_vector, [n_repmat 1]);

eqLatitude_vector = repmat(eqLatitude_vector, [n_repmat 1]);
eqLongitude_vector = repmat(eqLongitude_vector, [n_repmat 1]);
staLatitude_vector = repmat(staLatitude_vector, [n_repmat 1]);
staLongitude_vector = repmat(staLongitude_vector, [n_repmat 1]);  



n = length(log_PGA_vector);

%% SVD Input Parameters!!

method_SVD = 'Lanczos'; % Halko/Lanczos

% Parameters in Fast_SVD_Improved_function

tol_eval_truncation = 1E-8  ;
num_evals_taken = 200;
number_of_passes = 1; 


flag_GPU = 0;

%% Memory Requirements

[user,~] = memory; 
memory_used_initial = user.MemUsedMATLAB/1E9;

%% Plots Data!!


geoscatter(eqLatitude_vector, eqLongitude_vector)
hold on
geoscatter(staLatitude_vector, staLongitude_vector)
geobasemap colorterrain
legend('Earthquakes', 'Stations')
hold off

%% Defines input grid 1D functions, outside Likelihood function, Just once


% Lat, Lon grid

% Capture range of both eqLat/eqLon AND staLat/staLon- With some extra space

tlat_min = 30; tlat_max = 38; num_pts_tlat = 1000;
tlon_min = -123; tlon_max = -113; num_pts_tlon = 1000;


tlat_vector = linspace(tlat_min, tlat_max, num_pts_tlat);
tlon_vector = linspace(tlon_min, tlon_max, num_pts_tlon);

log_RJB_vector = log(sqrt(RJB_vector.^2 + h^2));
Mag_vector_2 = Mag_vector.^2;

% NORMALIZED VS30 and RJB vectors here

VS30_vector = VS30_vector/mean(VS30_vector);
RJB_vector = RJB_vector/mean(RJB_vector);



% 1D grids separated

xg_lat = {{ tlat_vector' }};    
xg_lon  = {{ tlon_vector' }};

cov_fun_lat = {{@covSEiso}}; 
cov_fun_lon = {{@covSEiso}}; 

covg_fun_lat = {@apxGrid, cov_fun_lat, xg_lat}; 
covg_fun_lon = {@apxGrid, cov_fun_lon, xg_lon}; 






sn = 0.5219;


%% Hyperparamters

% freq
% rho_eqX1a
% theta_eqX1a
% theta_statX1bi
% rho_statX1bii
% theta_statX1bii
% rho_cA
% theta_cA
% sigma_cA

Greg_Table = [0.3311311;...
              35.95735612;... %km
              0.101994683;...
              0.231812357;...
              12.96589957;... %km
              0.438331313;...
              47.09786233;... %km
              0.000679766;...
              1.00E-11];

% Term m1

ell_m1 = km2deg(Greg_Table(2, 1)); sf_m1 = Greg_Table(1, 1);

hyp_m1 = log([ell_m1 ; sf_m1]); % Input: log(lambda), Ouput: 1/lambda^2!!


% Term 0

ell_0 = (Greg_Table(4, 1)); sf_0 = Greg_Table(3, 1);

hyp_0 = log([ell_0 ; sf_0]);


tic
%% Updates 1D Grid Covariance Matrices with new hyperparameters

% Beta_m1

[covg_m1_lat, Mg_m1_lat] = feval(covg_fun_lat{:}, [hyp_m1(1) ; 1/2*hyp_m1(2)], eqLatitude_vector); % covSEiso 1D with sqrt(theta)!!
[covg_m1_lon, Mg_m1_lon] = feval(covg_fun_lon{:}, [hyp_m1(1) ; 1/2*hyp_m1(2)], eqLongitude_vector); % covSEiso 1D with sqrt(theta)!!

% Beta_0

[covg_0_lat, Mg_0_lat] = feval(covg_fun_lat{:}, [hyp_0(1) ; 1/2*hyp_0(2)], staLatitude_vector); % covSEiso 1D with sqrt(theta)!!
[covg_0_lon, Mg_0_lon] = feval(covg_fun_lon{:}, [hyp_0(1) ; 1/2*hyp_0(2)], staLongitude_vector); % covSEiso 1D with sqrt(theta)!!

%% Low-rank decomposition of each K term

% Beta_m1

[U_latlon_m1, D_latlon_m1] = get_SVD_K_SKIP_function(Mg_m1_lat, Mg_m1_lon, covg_m1_lat, covg_m1_lon, tol_eval_truncation, num_evals_taken, number_of_passes, flag_GPU, method_SVD);

% Beta_0

[U_latlon_0, D_latlon_0] = get_SVD_K_SKIP_function(Mg_0_lat, Mg_0_lon, covg_0_lat, covg_0_lon, tol_eval_truncation, num_evals_taken, number_of_passes, flag_GPU, method_SVD);

                                
%% Creates Fast function handles for each Non-Constant term

% Beta_m1

UD_latlon_m1 = U_latlon_m1.*D_latlon_m1'; % Pre-computed
Km1_mvm_SKIP_function = @(x) UD_latlon_m1 * (U_latlon_m1' * x);


% Beta_0

UD_latlon_0 = U_latlon_0.*D_latlon_0'; % Pre-computed
K0_mvm_SKIP_function = @(x) UD_latlon_0 * (U_latlon_0' * x);

                                     

                                                  
%% Creates Fast function handles for each Constant term

%%% Pi Beta_m1

% Kgm1_mvm_function_Const = @(x) exp(hyp_m1_Const(1))^2 * sum(x, 1) .* ones(size(x)); 

%%% Pi Beta_0

% Kg0_mvm_function_Const = @(x) exp(hyp_0_Const(1))^2  * sum(x, 1) .* ones(size(x)); 


%% Sum of fast functions


K_mvm_total_SKIP = @(x) Km1_mvm_SKIP_function(x) + ...
                        K0_mvm_SKIP_function(x);


K_function_over_sn2 = @(x) 1/sn^2 * K_mvm_total_SKIP(x);
                           
% In this way, more accurate efuns than when using B matrix (cf check_PseudoInverse)


[U_svd, D_svd_K, V_svd] = Fast_SVD_Improved_function(K_function_over_sn2, n, num_evals_taken, 2, method_SVD);


% Adds Id to 1/sn^2*K to get evals of: B = Id + 1/sn^2 * K 

D_svd_B = D_svd_K + ones(length(D_svd_K), 1);
                 

%%% Computes log(det(B_Matrix))

%ldB2_sparse = sum(log(real(D_svd_B)))/2;


%%% Computes low-rank decomposition of Pseudo-inverse of K

LHS_Matrix = U_svd.*(1/sn^2)./(real(D_svd_B))';  % K_total^{-1} = 1/sn^2 *B^{-1}
RHS_Matrix = U_svd;


%% Computes alpha_vector!!

%%% Computes alpha_vector using CG solver

tol = 1e-8 * 1; 

maxit_number_CG = 1E3;

K_function_with_sn2 = @(x) sn^2*x + K_mvm_total_SKIP(x);

y = log_PGA_vector;

[alpha_vector, num_iter_perfomed] = CG_Solver_function(K_function_with_sn2, y, tol, maxit_number_CG);


% time_SKIP = toc;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Imports Data for Prediction

n_data_pred = 100000;

eqLatitude_vector_Pred = eqLatitude_vector(1:n_data_pred);
eqLongitude_vector_Pred = eqLongitude_vector(1:n_data_pred);

staLatitude_vector_Pred = staLatitude_vector(1:n_data_pred);
staLongitude_vector_Pred = staLongitude_vector(1:n_data_pred);



%% Computes SKI for each term for Predictions


% Beta m1

[~, Mg_m1_lat_Pred] = apxGrid({{@covSEiso}}, {{ tlat_vector' }}, [hyp_m1(1) ; 1/2*hyp_m1(2)], eqLatitude_vector_Pred);
[~, Mg_m1_lon_Pred] = apxGrid({{@covSEiso}}, {{ tlon_vector' }}, [hyp_m1(1) ; 1/2*hyp_m1(2)], eqLongitude_vector_Pred);


% Beta 0

[~, Mg_0_lat_Pred] = apxGrid({{@covSEiso}}, {{ tlat_vector' }}, [hyp_0(1) ; 1/2*hyp_0(2)], staLatitude_vector_Pred);
[~, Mg_0_lon_Pred] = apxGrid({{@covSEiso}}, {{ tlon_vector' }}, [hyp_0(1) ; 1/2*hyp_0(2)], staLongitude_vector_Pred);





%% Low-rank decomposition of each term!!

% Beta m1

[U_m1_Pred, D_m1_Pred, V_m1_Pred] = get_SVD_Ks_Rectangular_SKIP_function(Mg_m1_lat, Mg_m1_lon, ...
                                                        covg_m1_lat, covg_m1_lon, ...
                                                        Mg_m1_lat_Pred, Mg_m1_lon_Pred, ...
                                                        tol_eval_truncation, num_evals_taken, number_of_passes, flag_GPU, method_SVD);

                                                    

% Beta 0 

[U_0_Pred, D_0_Pred, V_0_Pred] = get_SVD_Ks_Rectangular_SKIP_function(Mg_0_lat, Mg_0_lon, ...
                                                        covg_0_lat, covg_0_lon, ...
                                                        Mg_0_lat_Pred, Mg_0_lon_Pred, ...
                                                        tol_eval_truncation, num_evals_taken, number_of_passes, flag_GPU, method_SVD);
                                                    

                                                     
%% Creates function handles for each Non-Constant term
  

% Beta m1 for K_{X, X_Pred}

UD_m1_Pred = U_m1_Pred.*D_m1_Pred'; 

if n_data_pred <= n 

    Km1_mvm_Pred_SKIP_horizontal_function = @(x) UD_m1_Pred * (V_m1_Pred' * x);
    Km1_mvm_Pred_SKIP_vertical_function = @(x) V_m1_Pred  * (UD_m1_Pred' * x);

else
    
    Km1_mvm_Pred_SKIP_vertical_function = @(x) UD_m1_Pred * (V_m1_Pred' * x);
    Km1_mvm_Pred_SKIP_horizontal_function = @(x) V_m1_Pred  * (UD_m1_Pred' * x);
    
end



% Beta 0 for K_{X, X_Pred}

UD_0_Pred = U_0_Pred.*D_0_Pred'; 

if n_data_pred <= n 

    K0_mvm_Pred_SKIP_horizontal_function = @(x) UD_0_Pred * (V_0_Pred' * x);
    K0_mvm_Pred_SKIP_vertical_function = @(x) V_0_Pred  * (UD_0_Pred' * x);

else
    
    K0_mvm_Pred_SKIP_vertical_function = @(x) UD_0_Pred * (V_0_Pred' * x);
    K0_mvm_Pred_SKIP_horizontal_function = @(x) V_0_Pred  * (UD_0_Pred' * x);
    
end



%% Creates Fast function handles for each Constant term

% Pi Beta_1a

if n_data_pred <= n

    Km1_mvm_Pred_Const_horizontal_function = @(x) exp(hyp_m1_Const(1))^2 * sum(x, 1) .* ones(n_data_pred, 1); % Pi^2 * ones(n, n) * x!!
    Km1_mvm_Pred_Const_vertical_function = @(x) exp(hyp_m1_Const(1))^2 * sum(x, 1) .* ones(n, 1); % Pi^2 * ones(n, n) * x!!
    
    K0_mvm_Pred_Const_horizontal_function = @(x) exp(hyp_0_Const(1))^2 * sum(x, 1) .* ones(n_data_pred, 1); % Pi^2 * ones(n, n) * x!!
    K0_mvm_Pred_Const_vertical_function = @(x) exp(hyp_0_Const(1))^2 * sum(x, 1) .* ones(n, 1); % Pi^2 * ones(n, n) * x!!


else
    
    Km1_mvm_Pred_Const_vertical_function   = @(x) exp(hyp_m1_Const(1))^2 * sum(x, 1) .* ones(n_data_pred, 1); % Pi^2 * ones(n, n) * x!!
    Km1_mvm_Pred_Const_horizontal_function  = @(x) exp(hyp_m1_Const(1))^2 * sum(x, 1) .* ones(n, 1); % Pi^2 * ones(n, n) * x!!
    
    K0_mvm_Pred_Const_vertical_function     = @(x) exp(hyp_0_Const(1))^2 * sum(x, 1) .* ones(n_data_pred, 1); % Pi^2 * ones(n, n) * x!!
    K0_mvm_Pred_Const_horizontal_function    = @(x) exp(hyp_0_Const(1))^2 * sum(x, 1) .* ones(n, 1); % Pi^2 * ones(n, n) * x!!
    
    
end


%% Computes Singular Value Decomposition of K_{X, X*}
%%%
%%% K_{X, X*} = [U_svd_K_Pred * (D_svd_K_Pred)] * V_svd_K_Pred'
%%%


% Horizontal

K_mvm_SKIP_Pred_horizontal_total_function = @(x) Km1_mvm_Pred_SKIP_horizontal_function(x) + ...
                                                K0_mvm_Pred_SKIP_horizontal_function(x);
                                                

                             
% Vertical

K_mvm_SKIP_Pred_vertical_total_function = @(x) Km1_mvm_Pred_SKIP_vertical_function(x) + ...
                                                K0_mvm_Pred_SKIP_vertical_function(x);
                                 




% Rectangular SVD


num_cols = max(n, n_data_pred);

num_evals_taken_SVD = min(min(n, n_data_pred), num_evals_taken*1);

[U_svd_K_Pred, D_svd_K_Pred, V_svd_K_Pred] = Fast_SVD_Rectangular_function(K_mvm_SKIP_Pred_horizontal_total_function, ...
                                                                           K_mvm_SKIP_Pred_vertical_total_function, ...
                                                                           num_cols, num_evals_taken_SVD, 2, method_SVD);

% Singular Value Decomposition decomposition of K_{X, X*}:
% K_{X, X*} = [U_svd_K_Pred * (D_svd_K_Pred)] * V_svd_K_Pred'
%           = LHS_Matrix_Pred * RHS_Matrix_Pred'


if n_data_pred <= n

    LHS_Matrix_Pred = V_svd_K_Pred;   % n x d
    RHS_Matrix_Pred = U_svd_K_Pred.*(real(D_svd_K_Pred))'; % n_data_pred x d

else
    
    LHS_Matrix_Pred = U_svd_K_Pred.*(real(D_svd_K_Pred))'; % n x d
    RHS_Matrix_Pred = V_svd_K_Pred;    % n_data_pred x d

    
end



%% Computes Median Predictions
%%%
%%% K_{X, X*}  = LHS_Matrix_Pred * RHS_Matrix_Pred', so:
%%%
%%% K_{X*, X} = K_{X, X*}^T = RHS_Matrix_Pred*LHS_Matrix_Pred'
%%%
%%%
%%% Equation (10) from Landwehr:  
%%%
%%% mu = K_{X*, X} * (K_{X, X} + sn^2*Id)^{-1} * y 
%%%    = K_{X*, X} * alpha
%%%    = RHS_Matrix_Pred*LHS_Matrix_Pred' * alpha
%%%
%%% (no noise term in K_{X*, X} for predictions)
%%%      


% mu_vector = RHS_Matrix_Pred * (LHS_Matrix_Pred' * alpha_vector); 
        
mu_vector = K_mvm_SKIP_Pred_horizontal_total_function(alpha_vector);

%% Computes Epistemic Uncertainty
%%%
%%% Following Equation (11) from Landwher

%%% Self covariance diagonal vector K_{X*, X*}


kss_vector =  exp(hyp_m1(2))^2 * ones(n_data_pred, 1) +  ...
              exp(hyp_0(2))^2 * ones(n_data_pred, 1);
     


%%% Cross-covariance diagonal vector: 
%%% diag(K_{X*, X} * K_{X, X}^{-1}*K_{X, X*})


temp_Matrix_1 = RHS_Matrix_Pred *  ((LHS_Matrix_Pred' * LHS_Matrix) * ...
                                    (RHS_Matrix' * LHS_Matrix_Pred)); 


temp_Matrix_2 = RHS_Matrix_Pred';

ks_diag_vector = sum(temp_Matrix_1.*temp_Matrix_2', 2);


% Total epistemic uncertainty vector

Psi_vector  = kss_vector - ks_diag_vector;

time_SKIP = toc;

%% Checks computational time and memory used

% histogram(mu_vector - y(1:n_data_pred)) ; xlabel('log Median FAS residual');


[user,sys] = memory;
memory_used_SKIP = user.MemUsedMATLAB/1E9 - memory_used_initial;

disp(strcat('Time elapsed with SKIP is: ', {' '},  num2str(time_SKIP), ' s'))

disp(strcat('Memory used with SKIP approach is: ', {' '},  num2str(memory_used_SKIP), ' GB'))
