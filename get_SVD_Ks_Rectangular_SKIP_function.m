function [U_latlon, D_latlon, V_latlon] = get_SVD_Ks_Rectangular_SKIP_function(W_lat_x, W_lon_x, covg_lat, covg_lon, W_lat_z, W_lon_z, tol_eval_latlon, num_evals_taken, number_of_passes, flag_GPU, method)


%% Gets a low-rank decomposition of a rectangular covariance matrix K using SKIP with no linear term

%%% Get SVD of covariance matrix K of the form: K = K_lat .* K_lon,
%%% such that : K = U D V^T


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

    K_lat_LHS_mvm_function = @(x) W_lat_x * covg_lat.mvm(W_lat_z'*x);
    K_lat_RHS_mvm_function = @(x) W_lat_z * covg_lat.mvm(W_lat_x'*x);
    
    K_lon_LHS_mvm_function = @(x) W_lon_x * covg_lon.mvm(W_lon_z'*x);
    K_lon_RHS_mvm_function = @(x) W_lon_z * covg_lon.mvm(W_lon_x'*x);

else
    
    K_lat_RHS_mvm_function = @(x) W_lat_x * covg_lat.mvm(W_lat_z'*x);
    K_lat_LHS_mvm_function = @(x) W_lat_z * covg_lat.mvm(W_lat_x'*x);
    
    K_lon_RHS_mvm_function = @(x) W_lon_x * covg_lon.mvm(W_lon_z'*x);
    K_lon_LHS_mvm_function = @(x) W_lon_z * covg_lon.mvm(W_lon_x'*x);

end




%% Finds Evals/Efuns of K_lat and K_lon separately

% Input RHS matrix should be horizontal

[U_lat_rect, D_lat_rect, V_lat_rect] = Fast_SVD_Rectangular_function(K_lat_RHS_mvm_function, K_lat_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes, method);

% Input RHS matrix should be horizontal

[U_lon_rect, D_lon_rect, V_lon_rect] = Fast_SVD_Rectangular_function(K_lon_RHS_mvm_function, K_lon_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes, method);


indices_to_remove_lat = find(D_lat_rect < tol_eval_latlon);
indices_to_remove_lon = find(D_lon_rect < tol_eval_latlon);

D_lat_rect(indices_to_remove_lat) = [];
U_lat_rect(:, indices_to_remove_lat) = [];
V_lat_rect(:, indices_to_remove_lat) = [];

D_lon_rect(indices_to_remove_lon) = [];
U_lon_rect(:, indices_to_remove_lon) = [];
V_lon_rect(:, indices_to_remove_lon) = [];





%% Finds Evals/Efuns of Rect K_lat_lon


UD_lat_rect = U_lat_rect.*D_lat_rect';

UD_lon_rect = U_lon_rect.*D_lon_rect';

% Uses GPU

if flag_GPU == 1
    
    V_lat_rect = gpuArray(V_lat_rect);
    UD_lat_rect = gpuArray(UD_lat_rect);
    V_lon_rect = gpuArray(V_lon_rect);
    UD_lon_rect = gpuArray(UD_lon_rect);

end

% MVM function handles for rectangular matrices

K_latlon_horizontal_mvm_function = @(x) Delta_mvm_SKIP_Rectangular_faster_function(V_lat_rect, ...
                                                                UD_lat_rect, ...
                                                                V_lon_rect, ...
                                                                UD_lon_rect, ...
                                                                x);


K_latlon_vertical_mvm_function = @(x) Delta_mvm_SKIP_Rectangular_faster_function(UD_lat_rect, ...
                                                                V_lat_rect, ...
                                                                UD_lon_rect, ...
                                                                V_lon_rect, ...
                                                                x);





[U_latlon, D_latlon, V_latlon] = Fast_SVD_Rectangular_function(K_latlon_horizontal_mvm_function, K_latlon_vertical_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes, method);

indices_to_remove_latlon = find(D_latlon < tol_eval_latlon);

D_latlon(indices_to_remove_latlon) = [];
U_latlon(:, indices_to_remove_latlon) = [];
V_latlon(:, indices_to_remove_latlon) = [];

%% Removes zeros efuns

U_latlon = U_latlon(1:num_rows, :);


%% Re-attributes appropriate U and V

if n_data_pred > n
    
    temp_U = U_latlon;
    
    U_latlon = V_latlon;
    V_latlon = temp_U;
    

end
