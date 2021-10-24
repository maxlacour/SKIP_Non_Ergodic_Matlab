function [U_latlon, D_latlon, V_latlon] = get_Low_Rank_Ks_SKIP_function(Mg_lat_x, Mg_lon_x, covg_lat, covg_lon, Mg_lat_z, Mg_lon_z, tol_eval_latlon, num_evals_taken, number_of_passes)


n = size(Mg_lat_x, 1);

n_data_pred = size(Mg_lat_z, 1);

% Size of input rectangular matrices!!
num_rows = min(n, n_data_pred);
num_cols = max(n, n_data_pred);

% Number of evals taken should not be greater than size of matrix!!

if num_evals_taken > num_rows
    num_evals_taken = num_rows;
end

%% Low-rank decomposition of each term!!

if n_data_pred <= n

    K_lat_LHS_mvm_function = @(x) Mg_lat_x * covg_lat.mvm(Mg_lat_z'*x);
    K_lat_RHS_mvm_function = @(x) Mg_lat_z * covg_lat.mvm(Mg_lat_x'*x);
    
    K_lon_LHS_mvm_function = @(x) Mg_lon_x * covg_lon.mvm(Mg_lon_z'*x);
    K_lon_RHS_mvm_function = @(x) Mg_lon_z * covg_lon.mvm(Mg_lon_x'*x);

else
    
    K_lat_RHS_mvm_function = @(x) Mg_lat_x * covg_lat.mvm(Mg_lat_z'*x);
    K_lat_LHS_mvm_function = @(x) Mg_lat_z * covg_lat.mvm(Mg_lat_x'*x);
    
    K_lon_RHS_mvm_function = @(x) Mg_lon_x * covg_lon.mvm(Mg_lon_z'*x);
    K_lon_LHS_mvm_function = @(x) Mg_lon_z * covg_lon.mvm(Mg_lon_x'*x);

end

%% Finds Evals/Efuns of K_lat and K_lon separately!!

% Input RHS matrix should be horizontal!!

[U_lat_rect, D_lat_rect, V_lat_rect] = Fast_SVD_Rectangular_Improved_function(K_lat_RHS_mvm_function, K_lat_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);

% Input RHS matrix should be horizontal!!

[U_lon_rect, D_lon_rect, V_lon_rect] = Fast_SVD_Rectangular_Improved_function(K_lon_RHS_mvm_function, K_lon_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);


indices_to_remove_lat = find(D_lat_rect < tol_eval_latlon);
indices_to_remove_lon = find(D_lon_rect < tol_eval_latlon);

D_lat_rect(indices_to_remove_lat) = [];
U_lat_rect(:, indices_to_remove_lat) = [];
V_lat_rect(:, indices_to_remove_lat) = [];

D_lon_rect(indices_to_remove_lon) = [];
U_lon_rect(:, indices_to_remove_lon) = [];
V_lon_rect(:, indices_to_remove_lon) = [];



%% Extends efuns with zeros for merging!!

% Non-zeros are placed horizontally!!

U_lat_rect_extended = [U_lat_rect ; zeros(num_cols - num_rows, size(U_lat_rect, 2))];
U_lon_rect_extended = [U_lon_rect ; zeros(num_cols - num_rows, size(U_lon_rect, 2))];


%% Finds Evals/Efuns of Rect K_lat_lon!!


UD_lat_rect_extended = U_lat_rect_extended.*D_lat_rect';

UD_lon_rect_extended = U_lon_rect_extended.*D_lon_rect';

K_latlon_RHS_mvm_function = @(x) Delta_mvm_SKIP_faster_function(V_lat_rect, ...
                                                                UD_lat_rect_extended, ...
                                                                UD_lon_rect_extended, ...
                                                                V_lon_rect, ...
                                                                x);


K_latlon_LHS_mvm_function = @(x) Delta_mvm_SKIP_faster_function(UD_lat_rect_extended, ...
                                                                V_lat_rect, ...
                                                                V_lon_rect, ...
                                                                UD_lon_rect_extended, ...
                                                                x);



% Input RHS matrix should be horizontal!!

[U_latlon, D_latlon, V_latlon] = Fast_SVD_Rectangular_Improved_function(K_latlon_RHS_mvm_function, K_latlon_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);

indices_to_remove_latlon = find(D_latlon < tol_eval_latlon);

D_latlon(indices_to_remove_latlon) = [];
U_latlon(:, indices_to_remove_latlon) = [];
V_latlon(:, indices_to_remove_latlon) = [];

%% Removes zeros efuns!!

U_latlon = U_latlon(1:num_rows, :);


%% Re-attributes appropriate U and V!!

if n_data_pred > n
    
    temp_U = U_latlon;
    
    U_latlon = V_latlon;
    V_latlon = temp_U;
    

end