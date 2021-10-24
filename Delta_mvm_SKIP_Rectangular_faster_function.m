function y = Delta_mvm_SKIP_Rectangular_faster_function(U_Matrix_LHS, UD_Matrix_LHS, U_Matrix_RHS, UD_Matrix_RHS, vector_input)


%% Performs computation of: (UD_Matrix_LHS * U_Matrix_LHS') * vector_input * (UD_Matrix_RHS * U_Matrix_RHS')
% Equation (10) from SKIP with rectangular matrices


v_LHS = size(U_Matrix_LHS, 2);
v_RHS = size(U_Matrix_RHS, 2);

[n, v] = size(vector_input);

num_row = size(UD_Matrix_LHS, 1);

%%

y = bsxfun(@times, U_Matrix_LHS, permute(vector_input,[1,3,2])); % U_Matrix_LHS times vec

y = reshape(reshape(y, n*v_LHS, []), n, []); % Turns into matrix, no transpose yet because it's expensive

% Truncates rectangle
y = U_Matrix_RHS' * y; % times UD_Matrix_RHS



y = reshape(permute(reshape(y, v_RHS, [], v),  [1, 3, 2]), [], v_LHS)'; %
y = UD_Matrix_LHS * y; % times UD_Matrix_LHS



y = reshape(permute(reshape(y, 1, []),[1,3,2]), [],v);
y = y .* reshape(UD_Matrix_RHS, 1, [])'; % Element-wise product

y = reshape(permute(y, [1 3 2]), num_row, [], v); 
y = reshape(sum(y, 2), num_row, v);

end