function [U_linearlatlon, D_linearlatlon] = get_Low_Rank_K_Linear_SKIP_function(Mg_lat, Mg_lon, covg_lat, covg_lon, linear_vector, tol_eval_latlon, num_evals_taken, number_of_passes)

%% Gets a low-rank decomposition of matrix covariance K using SKIP with one merge only, and no linear terms!!

% Only with a covariance matrix K of the form: K = K_lat .* K_lon

%% Creates K MVM functions!!

n = size(Mg_lat, 1);


K_lat_mvm_function = @(x) Mg_lat * covg_lat.mvm(Mg_lat'*x);
K_lon_mvm_function = @(x) Mg_lon * covg_lon.mvm(Mg_lon'*x);

%% Finds Evals/Efuns of K_lat and K_lon separately!!

[U_lat, D_lat, ~] = Fast_SVD_Improved_function(K_lat_mvm_function, n, num_evals_taken, number_of_passes);
[U_lon, D_lon, ~] = Fast_SVD_Improved_function(K_lon_mvm_function, n, num_evals_taken, number_of_passes);

indices_to_remove_lat = find(D_lat < tol_eval_latlon);
indices_to_remove_lon = find(D_lon < tol_eval_latlon);

D_lat(indices_to_remove_lat) = [];
D_lon(indices_to_remove_lon) = [];
U_lat(:, indices_to_remove_lat) = [];
U_lon(:, indices_to_remove_lon) = [];



%% Finds Evals/Efuns of K_linear o K_lat!!

%%% Without Merge!!

U_linearlat = U_lat.*linear_vector;
norm_U_linearlat = vecnorm(U_linearlat);
U_linearlat = U_linearlat./norm_U_linearlat;
D_linearlat = D_lat.*norm_U_linearlat'.^2;

[~, d_ind] = sort(D_linearlat, 'descend');
D_linearlat = D_linearlat(d_ind);
U_linearlat = U_linearlat(:, d_ind);

%% Finds Evals/Efuns of K_lat_lon!!

UD_linearlat = U_linearlat.*D_linearlat'; % Pre-computed!!
UD_lon = U_lon.*D_lon';

K_linearlatlon_mvm_function = @(x) Delta_mvm_SKIP_faster_function(U_linearlat, UD_linearlat, ...
                                                            U_lon, ...
                                                            UD_lon, ...
                                                            x);



[U_linearlatlon, D_linearlatlon, ~] = Fast_SVD_Improved_function(K_linearlatlon_mvm_function, n, num_evals_taken, number_of_passes);

indices_to_remove_linearlatlon = find(D_linearlatlon < tol_eval_latlon);

D_linearlatlon(indices_to_remove_linearlatlon) = [];
U_linearlatlon(:, indices_to_remove_linearlatlon) = [];


end                                
