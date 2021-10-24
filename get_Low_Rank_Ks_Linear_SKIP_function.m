function [U_linearlatlon, D_linearlatlon, V_linearlatlon] = get_Low_Rank_Ks_Linear_SKIP_function(Mg_lat_x, Mg_lon_x, covg_lat, covg_lon, Mg_lat_z, Mg_lon_z, linear_vector_x, linear_vector_z, tol_eval_latlon, num_evals_taken, number_of_passes)


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


%% Finds Evals/Efuns of K_linear o K_lat!!

if length(linear_vector_x) < length(linear_vector_z)
    
    linear_vector_LHS = linear_vector_x;
    linear_vector_RHS = linear_vector_z;
    
else
    
    linear_vector_RHS = linear_vector_x;
    linear_vector_LHS = linear_vector_z;

end

%%% Without Merge!!

U_linearlat_rect = U_lat_rect.*linear_vector_LHS;
V_linearlat_rect = V_lat_rect.*linear_vector_RHS;


norm_U_linearlat = vecnorm(U_linearlat_rect);
norm_V_linearlat = vecnorm(V_linearlat_rect);

U_linearlat_rect = U_linearlat_rect./norm_U_linearlat;
V_linearlat_rect = V_linearlat_rect./norm_V_linearlat;

D_linearlat_rect = (D_lat_rect.*norm_U_linearlat').*norm_V_linearlat';

[~, d_ind] = sort(D_linearlat_rect, 'descend');

D_linearlat_rect = D_linearlat_rect(d_ind);
U_linearlat_rect = U_linearlat_rect(:, d_ind);
V_linearlat_rect = V_linearlat_rect(:, d_ind);

%% Extends efuns with zeros for merging!!

% Non-zeros are placed horizontally!!

U_linearlat_rect_extended = [U_linearlat_rect ; zeros(num_cols - num_rows, size(U_linearlat_rect, 2))];
U_lon_rect_extended = [U_lon_rect ; zeros(num_cols - num_rows, size(U_lon_rect, 2))];


%% Finds Evals/Efuns of Rect K_lat_lon!!


UD_linearlat_rect_extended = U_linearlat_rect_extended.*D_linearlat_rect';

UD_lon_rect_extended = U_lon_rect_extended.*D_lon_rect';

K_linearlatlon_RHS_mvm_function = @(x) Delta_mvm_SKIP_faster_function(V_linearlat_rect, ...
                                                                UD_linearlat_rect_extended, ...
                                                                UD_lon_rect_extended, ...
                                                                V_lon_rect, ...
                                                                x);


K_linearlatlon_LHS_mvm_function = @(x) Delta_mvm_SKIP_faster_function(UD_linearlat_rect_extended, ...
                                                                V_linearlat_rect, ...
                                                                V_lon_rect, ...
                                                                UD_lon_rect_extended, ...
                                                                x);



% Input RHS matrix should be horizontal!!

[U_linearlatlon, D_linearlatlon, V_linearlatlon] = Fast_SVD_Rectangular_Improved_function(K_linearlatlon_RHS_mvm_function, K_linearlatlon_LHS_mvm_function, ...
    num_cols, num_evals_taken, number_of_passes);

indices_to_remove_linearlatlon = find(D_linearlatlon < tol_eval_latlon);

D_linearlatlon(indices_to_remove_linearlatlon) = [];
U_linearlatlon(:, indices_to_remove_linearlatlon) = [];
V_linearlatlon(:, indices_to_remove_linearlatlon) = [];

%% Removes zeros efuns!!

U_linearlatlon = U_linearlatlon(1:num_rows, :);


%% Re-attributes appropriate U and V!!

if n_data_pred > n
    
    temp_U = U_linearlatlon;
    
    U_linearlatlon = V_linearlatlon;
    V_linearlatlon = temp_U;
    

end