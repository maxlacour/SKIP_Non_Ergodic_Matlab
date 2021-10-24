function [U_dK_ell, D_dK_ell, V_dK_ell] = get_Low_Rank_dKell_function(U_eq, D_eq, V_eq, U_latlon, D_latlon, tol_eval_truncation, num_evals_taken, number_of_passes)

%% Applies merge of K with Norm_Matrix!!

n = size(U_eq, 1);

% Beta_m1

UD_m1 = U_latlon.*D_latlon'; % Pre-computed!!
UD_eq_latlon = V_eq.*D_eq'; % Norm_Matrix not positive-definite!!


% SKIP mvm function

dK_ell_mvm_function = @(x) Delta_mvm_SKIP_faster_function(U_latlon, ...
                                                          UD_m1, ...
                                                          U_eq, ...
                                                          UD_eq_latlon, ...
                                                          x);

% Takes most of the computation, because Delta_mvm_SKIP_faster_function
% takes care of the Element-wise MVM products!!

[U_dK_ell, D_dK_ell, V_dK_ell] = Fast_SVD_Improved_function(dK_ell_mvm_function, n, num_evals_taken, number_of_passes);

indices_to_remove_dK_ell = find(D_dK_ell < tol_eval_truncation);

D_dK_ell(indices_to_remove_dK_ell) = [];
U_dK_ell(:, indices_to_remove_dK_ell) = [];
V_dK_ell(:, indices_to_remove_dK_ell) = []; % Norm_Matrix and resulting Matrix are not positive definite!!!!




end