function y = Delta_mvm_SKIP_faster_function(U_Matrix_LHS, UD_Matrix_LHS, U_Matrix_RHS, UD_Matrix_RHS, vector_input)

%% Performs computation of: (UD_Matrix_LHS * U_Matrix_LHS') * vector_input * (UD_Matrix_RHS * U_Matrix_RHS')
% Equation (10) from SKIP

[n, v] = size(vector_input);

v_LHS = size(U_Matrix_LHS, 2);
v_RHS = size(U_Matrix_RHS, 2);

%% Multiple variables, expensive in RAM!! y_temp_1 up to 8 GB!!

% y_temp_1 = bsxfun(@times, U_Matrix_LHS, permute(vector_input,[1,3,2])); % U_Matrix_LHS times vec
% y_temp_2 = reshape(reshape(y_temp_1, n*v_LHS, []), n, []); % Turns into matrix, no transpose because it's expensive!!
% y_temp_3 = UD_Matrix_RHS' * y_temp_2; % times UD_Matrix_RHS
% y_temp_4 =  reshape(permute(reshape(y_temp_3, v_RHS, [], v),  [1, 3, 2]), [], v_LHS)'; %
% y_temp_5 = UD_Matrix_LHS * y_temp_4; % times UD_Matrix_LHS
% y_temp_5 = reshape(permute(reshape(y_temp_5, 1, []),[1,3,2]), [],v);
% y_temp_6 = y_temp_5 .* reshape(U_Matrix_RHS, 1, [])'; % Element-wise product!!
% y_temp_7 = reshape(permute(y_temp_6, [1 3 2]), n, [], v); 

%% One variable only!!

y = bsxfun(@times, U_Matrix_LHS, permute(vector_input,[1,3,2])); % U_Matrix_LHS times vec
% n * k * v



y = reshape(reshape(y, n*v_LHS, []), n, []); % Turns into matrix, no transpose yet because it's expensive!!
y = UD_Matrix_RHS' * y; % times UD_Matrix_RHS
% k * (k x v)



y = reshape(permute(reshape(y, v_RHS, [], v),  [1, 3, 2]), [], v_LHS)'; %
y = UD_Matrix_LHS * y; % times UD_Matrix_LHS
% n * (k * v)



y = reshape(permute(reshape(y, 1, []),[1,3,2]), [],v);
y = y .* reshape(U_Matrix_RHS, 1, [])'; % Element-wise product!!
% (n*k) * v

y = reshape(permute(y, [1 3 2]), n, [], v); 
y = reshape(sum(y, 2), n, v);
% n*v

%% One command only!!

% y = reshape(sum(...
%                     reshape(permute(...
%                     reshape(permute(reshape(...
%                     UD_Matrix_LHS * reshape(permute(reshape(UD_Matrix_RHS' * ...
%                     reshape(reshape(bsxfun(@times, U_Matrix_LHS, permute(vector_input,[1,3,2]))...
%                     , n*v_LHS, []), n, [])...
%                     , v_RHS, [], v),  [1, 3, 2]), [], v_LHS)'...
%                     , 1, []),[1,3,2]), [],v)...
%                     .* reshape(U_Matrix_RHS, 1, [])'...
%                     , [1 3 2]), n, [], v)...
%                     , 2), n, v);
                   
end