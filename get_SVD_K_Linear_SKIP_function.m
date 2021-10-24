function [U_linearlatlon, D_linearlatlon] = get_SVD_K_Linear_SKIP_function(W_lat, W_lon, covg_lat, covg_lon, linear_vector, tol_eval_latlon, num_evals_taken, number_of_passes, flag_GPU)

%% Gets a low-rank decomposition of covariance matrix K using SKIP with linear term

%%% Get SVD of covariance matrix K of the form: K = (X * X^T) .* K_lat .* K_lon,
%%% where X is the linear term, such that : K = U D V^T

% K = U * U^T
% If you want: K .* (X * X^T)
% Just multiply U with X
%
% (U*X) (U*X)^T = (X*X^T) * U*U^T
%% Creates K MVM functions

n = size(W_lat, 1);


K_lat_mvm_function = @(x) W_lat * covg_lat.mvm(W_lat'*x);
K_lon_mvm_function = @(x) W_lon * covg_lon.mvm(W_lon'*x);

%% Finds Evals/Efuns of K_lat and K_lon separately

[U_lat, D_lat, ~] = Fast_SVD_function(K_lat_mvm_function, n, num_evals_taken, number_of_passes);
[U_lon, D_lon, ~] = Fast_SVD_function(K_lon_mvm_function, n, num_evals_taken, number_of_passes);

indices_to_remove_lat = find(D_lat < tol_eval_latlon);
indices_to_remove_lon = find(D_lon < tol_eval_latlon);

D_lat(indices_to_remove_lat) = [];
D_lon(indices_to_remove_lon) = [];
U_lat(:, indices_to_remove_lat) = [];
U_lon(:, indices_to_remove_lon) = [];



%% Finds Evals/Efuns of K_linear o K_lat

%%% Avoids Merge with linear term

U_linearlat = U_lat.*linear_vector;
norm_U_linearlat = vecnorm(U_linearlat);
U_linearlat = U_linearlat./norm_U_linearlat;
D_linearlat = D_lat.*norm_U_linearlat'.^2;

[~, d_ind] = sort(D_linearlat, 'descend');
D_linearlat = D_linearlat(d_ind);
U_linearlat = U_linearlat(:, d_ind);

%% Finds Evals/Efuns of K_lat_lon

UD_linearlat = U_linearlat.*D_linearlat'; 
UD_lon = U_lon.*D_lon';

% Uses GPU

if flag_GPU == 1
    U_linearlat = gpuArray(U_linearlat);
    UD_linearlat = gpuArray(UD_linearlat);
    U_lon = gpuArray(U_lon);
    UD_lon = gpuArray(UD_lon);
    
end

K_linearlatlon_mvm_function = @(x) Delta_mvm_SKIP_faster_function(U_linearlat, ...
                                                                  UD_linearlat, ...
                                                                  U_lon, ...
                                                                  UD_lon, ...
                                                                  x);



[U_linearlatlon, D_linearlatlon, ~] = Fast_SVD_function(K_linearlatlon_mvm_function, n, num_evals_taken, number_of_passes);

indices_to_remove_linearlatlon = find(D_linearlatlon < tol_eval_latlon);

D_linearlatlon(indices_to_remove_linearlatlon) = [];
U_linearlatlon(:, indices_to_remove_linearlatlon) = [];


end                                
