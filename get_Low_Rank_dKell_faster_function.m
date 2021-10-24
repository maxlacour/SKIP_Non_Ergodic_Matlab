function [U_dK_ell, D_dK_ell, V_dK_ell] = get_Low_Rank_dKell_faster_function(U_eq, D_eq, V_eq, U_latlon, D_latlon)

%% Applies merge of K with Norm_Matrix!!

n = size(U_eq, 1);


UD_eq = V_eq.*D_eq'; % Norm_Matrix not positive-definite!!

linear_vector_LHS = UD_eq(:, 1:4);
linear_vector_RHS = U_eq(:, 1:4);

U_dK_ell = [U_latlon.*linear_vector_LHS(:, 1) U_latlon.*linear_vector_LHS(:, 2) ...
               U_latlon.*linear_vector_LHS(:, 3) U_latlon.*linear_vector_LHS(:, 4)];
           
V_dK_ell = [U_latlon.*linear_vector_RHS(:, 1) U_latlon.*linear_vector_RHS(:, 2) ...
            U_latlon.*linear_vector_RHS(:, 3) U_latlon.*linear_vector_RHS(:, 4)];
           
           

D_dK_ell = [D_latlon; D_latlon;
               D_latlon; D_latlon];
end