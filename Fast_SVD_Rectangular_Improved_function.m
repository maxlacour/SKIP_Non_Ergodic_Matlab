function [U_sorted, D_sorted, V_sorted] = Fast_SVD_Rectangular_Improved_function(K_RHS, K_LHS, n, k, number_of_passes)


%%% Input RHS matrix should be horizontal!!
%%% Input LHS matrix should be vertical!!


%%% Output matrix is horizontal!!

%%% Fast SVD decompisition of total covariance matrix!!
%%% Using Algorithm 5.1!!

% n is Size of full covariance matrix!!
% k is the target rank of the matrix!!

% Result is: K = U_sorted * D_sorted * V_sorted'; !!
       
% Do 1 pass by default, unless specifying 2 passes!!

if nargin<5
    number_of_passes = 1;
end



%% Stage A

% Given an m * n matrix A and an integer l, this scheme computes an m * l
% orthonormal matrix Q whose range approximates the range of A.


% Step 1: draw an n*l Gaussian random matrix Omega


p = 20; % p should be taken between 5 and 10!!

% l_value = min(k + p + 10, n);
l_value = min(k + p, n);


Omega_matrix = normrnd(0, 1, n, l_value);


if isnumeric(K_RHS) 
    Y_matrix = K_RHS * Omega_matrix;
else    
    Y_matrix = K_RHS(Omega_matrix);
end



[Q_0_matrix, ~] = qr(Y_matrix, 0);



%% Adds steps from Algorithm 4.4 here!! Given A = A*!!

% 1)

if isnumeric(K_LHS) 
    Y_1_tild_Matrix = K_LHS * Q_0_matrix;
else    
    Y_1_tild_Matrix = K_LHS(Q_0_matrix);
end


[Q_1_tild_matrix, ~] = qr(Y_1_tild_Matrix, 0);


% 2)

if isnumeric(K_RHS) 
    Y_1_Matrix = K_RHS * Q_1_tild_matrix;
else    
    Y_1_Matrix = K_RHS(Q_1_tild_matrix);
end


[Q_1_matrix, ~] = qr(Y_1_Matrix, 0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if number_of_passes == 2
    
% Once again!!


% 1)

        if isnumeric(K_LHS) 
            Y_2_tild_Matrix = K_LHS * Q_1_matrix;
        else    
            Y_2_tild_Matrix = K_LHS(Q_1_matrix);
        end


        [Q_2_tild_matrix, ~] = qr(Y_2_tild_Matrix, 0);


        % 2)

        if isnumeric(K_RHS) 
            Y_2_Matrix = K_RHS * Q_2_tild_matrix;
        else    
            Y_2_Matrix = K_RHS(Q_2_tild_matrix);
        end


        [Q_2_matrix, ~] = qr(Y_2_Matrix, 0);

        
        
end % End second pass!!

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Once again!!
% 
% 
% % 1)
% 
% if isnumeric(K) 
%     Y_3_tild_Matrix = K * Q_2_matrix;
% else    
%     Y_3_tild_Matrix = K(Q_2_matrix);
% end
% 
% 
% [Q_3_tild_matrix, ~] = qr(Y_3_tild_Matrix, 0);
% 
% 
% % 2)
% 
% if isnumeric(K) 
%     Y_3_Matrix = K * Q_3_tild_matrix;
% else    
%     Y_3_Matrix = K(Q_3_tild_matrix);
% end
% 
% 
% [Q_3_matrix, ~] = qr(Y_3_Matrix, 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Final Q_matrix

if number_of_passes == 1

    Q_matrix = Q_1_matrix;

else
    
    Q_matrix = Q_2_matrix;

end
%% Stage B

% Algorithm 5.1: Direct SVD Decomposition

% Given matrices A and Q such that (5.1) holds, this procedure computes an
% approximate factorization A ? U?V ?, where U and V are orthonormal,
% and ? is a nonnegative diagonal matrix.


if isnumeric(K_LHS) 
    B_matrix =  (K_LHS * Q_matrix)';  
else
    B_matrix = (K_LHS(Q_matrix))';  
end

[U_tild, D_small, V_reconstructed] = svd(B_matrix, 'econ'); % Computes eval on smaller matrix!!
%[U_tild, D_small, V_reconstructed] = svd(B_matrix, 0); % Computes eval on smaller matrix!!


% Reconstructs eigenvectors 


U_reconstructed = Q_matrix * U_tild;
%V_reconstructed = V_reconstructed; % V IS ALREADY GIVEN IN SMALL SVD!!!!

% Sorts singular values and vectors

[D_sorted, indices_eval] = sort(diag(D_small(1:k, 1:k)), 'descend');

U_sorted = U_reconstructed(:, indices_eval);
V_sorted = V_reconstructed(:, indices_eval);



end
