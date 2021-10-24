function [K_latlon_mvm_function] = get_Delta_MVM_K_SKIP_function(W_lat, W_lon, covg_lat, covg_lon, tol_eval_latlon, num_evals_taken, number_of_passes)

%% Gets a low-rank decomposition of matrix covariance K using SKIP with one merge only, and no linear terms

% Only with a covariance matrix K of the form: K = K_lat .* K_lon

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




%% Finds Evals/Efuns of K_lat_lon

UD_lat = U_lat.*D_lat';
UD_lon = U_lon.*D_lon';

K_latlon_mvm_function = @(x) Delta_mvm_SKIP_faster_function(U_lat, ...
                                                            UD_lat, ...
                                                            U_lon, ...
                                                            UD_lon, ...
                                                            x);


end                                
