function [K_linearlatlon_horizontal_mvm_function, K_linearlatlon_vertical_mvm_function] = get_Delta_MVM_Ks_Linear_SKIP_function(W_lat_x, W_lon_x, covg_lat, covg_lon, W_lat_z, W_lon_z, linear_vector_x, linear_vector_z, tol_eval_latlon, num_evals_taken, number_of_passes)


n = size(W_lat_x, 1);

n_data_pred = size(W_lat_z, 1);

% Size of input rectangular matrices

num_rows = min(n, n_data_pred);
num_cols = max(n, n_data_pred);

% Number of evals taken should not be greater than size of matrix

if num_evals_taken > num_rows
    num_evals_taken = num_rows;
end

%% Low-rank decomposition of each term

if n_data_pred <= n

    K_lat_vertical_mvm_function = @(x) W_lat_x * covg_lat.mvm(W_lat_z'*x);
    K_lat_horizontal_mvm_function = @(x) W_lat_z * covg_lat.mvm(W_lat_x'*x);
    
    K_lon_horizontal_mvm_function = @(x) W_lon_x * covg_lon.mvm(W_lon_z'*x);
    K_lon_vertical_mvm_function = @(x) W_lon_z * covg_lon.mvm(W_lon_x'*x);

else
    
    K_lat_horizontal_mvm_function = @(x) W_lat_x * covg_lat.mvm(W_lat_z'*x);
    K_lat_vertical_mvm_function = @(x) W_lat_z * covg_lat.mvm(W_lat_x'*x);
    
    K_lon_vertical_mvm_function = @(x) W_lon_x * covg_lon.mvm(W_lon_z'*x);
    K_lon_horizontal_mvm_function = @(x) W_lon_z * covg_lon.mvm(W_lon_x'*x);

end

%% Finds Evals/Efuns of K_lat and K_lon separately

% Input matrix should be horizontal

[U_lat_rect, D_lat_rect, V_lat_rect] = Fast_SVD_Rectangular_function(K_lat_horizontal_mvm_function, K_lat_vertical_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);

% Input matrix should be horizontal

[U_lon_rect, D_lon_rect, V_lon_rect] = Fast_SVD_Rectangular_function(K_lon_vertical_mvm_function, K_lon_horizontal_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);


indices_to_remove_lat = find(D_lat_rect < tol_eval_latlon);
indices_to_remove_lon = find(D_lon_rect < tol_eval_latlon);

D_lat_rect(indices_to_remove_lat) = [];
U_lat_rect(:, indices_to_remove_lat) = [];
V_lat_rect(:, indices_to_remove_lat) = [];

D_lon_rect(indices_to_remove_lon) = [];
U_lon_rect(:, indices_to_remove_lon) = [];
V_lon_rect(:, indices_to_remove_lon) = [];


%% Finds Evals/Efuns of K_linear o K_lat

if length(linear_vector_x) < length(linear_vector_z)
    
    linear_vector_vertical = linear_vector_x;
    linear_vector_horizontal = linear_vector_z;
    
else
    
    linear_vector_horizontal = linear_vector_x;
    linear_vector_vertical = linear_vector_z;

end

%%% Avoids Merge with linear term

U_linearlat_rect = U_lat_rect.*linear_vector_vertical;
V_linearlat_rect = V_lat_rect.*linear_vector_horizontal;


norm_U_linearlat = vecnorm(U_linearlat_rect);
norm_V_linearlat = vecnorm(V_linearlat_rect);

U_linearlat_rect = U_linearlat_rect./norm_U_linearlat;
V_linearlat_rect = V_linearlat_rect./norm_V_linearlat;

D_linearlat_rect = (D_lat_rect.*norm_U_linearlat').*norm_V_linearlat';

[~, d_ind] = sort(D_linearlat_rect, 'descend');

D_linearlat_rect = D_linearlat_rect(d_ind);
U_linearlat_rect = U_linearlat_rect(:, d_ind);
V_linearlat_rect = V_linearlat_rect(:, d_ind);



%% Finds Evals/Efuns of Rect K_lat_lon


UD_linearlat_rect = U_linearlat_rect.*D_linearlat_rect';

UD_lon_rect = U_lon_rect.*D_lon_rect';


% Uses GPU!!

if flag_GPU == 1

    V_linearlat_rect = gpuArray(V_linearlat_rect);
    UD_linearlat_rect = gpuArray(UD_linearlat_rect);
    V_lon_rect = gpuArray(V_lon_rect);
    UD_lon_rect = gpuArray(UD_lon_rect);

end


K_linearlatlon_horizontal_mvm_function = @(x) Delta_mvm_SKIP_Rectangular_faster_function(V_linearlat_rect, ...
                                                                UD_linearlat_rect, ...
                                                                V_lon_rect, ...
                                                                UD_lon_rect, ...
                                                                x);


K_linearlatlon_vertical_mvm_function = @(x) Delta_mvm_SKIP_Rectangular_faster_function(UD_linearlat_rect, ...
                                                                V_linearlat_rect, ...
                                                                UD_lon_rect, ...
                                                                V_lon_rect, ...
                                                                x);




end