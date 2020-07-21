%% Computes full covariance matrices!!

tic

% Beta m1

K_Beta_m1_Full = zeros(n, n);

for i = 1:n
    for j = 1:n
        
        K_Beta_m1_Full(i, j) = sf_m1^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector(j))^2)/(2*ell_m1^2));
        
        
    end
end




% K_SKIP_m1 = U_latlon_m1 * diag(D_latlon_m1) * U_latlon_m1';
% max(max(abs(K_SKIP_m1 - K_Beta_m1_Full)))

% Beta 0

K_Beta_0_Full = zeros(n, n);

for i = 1:n
    for j = 1:n
        
        K_Beta_0_Full(i, j) = sf_0^2 * exp(-((staLatitude_vector(i) - staLatitude_vector(j))^2 + ...
                                             (staLongitude_vector(i) - staLongitude_vector(j))^2)/(2*ell_0^2));
        
        
    end
end


% K_SKIP_0 = U_latlon_0 * diag(D_latlon_0) * U_latlon_0';
% max(max(abs(K_SKIP_0 - K_Beta_0_Full)))


% Beta 3


K_Beta_3_Full = zeros(n, n);

for i = 1:n
    for j = 1:n
        
        K_Beta_3_Full(i, j) = log_RJB_vector(i) * log_RJB_vector(j) * ...
                              sf_3^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector(j))^2)/(2*ell_3^2));
        
        
    end
end


% K_SKIP_3 = U_latlon_3 * diag(D_latlon_3) * U_latlon_3';
% max(max(abs(K_SKIP_3 - K_Beta_3_Full)))


% Beta 5


K_Beta_5_Full = zeros(n, n);

for i = 1:n
    for j = 1:n
        
        K_Beta_5_Full(i, j) = RJB_vector(i) * RJB_vector(j) * ...
                              sf_5^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector(j))^2)/(2*ell_5^2));
        
        
    end
end

% K_SKIP_5 = U_latlon_5 * diag(D_latlon_5) * U_latlon_5';
% max(max(abs(K_SKIP_5 - K_Beta_5_Full)))



% Beta 6 

K_Beta_6_Full = zeros(n, n);

for i = 1:n
    for j = 1:n
        
        K_Beta_6_Full(i, j) = VS30_vector(i) * VS30_vector(j) * ...
                              sf_6^2 * exp(-((staLatitude_vector(i) - staLatitude_vector(j))^2 + ...
                                             (staLongitude_vector(i) - staLongitude_vector(j))^2)/(2*ell_6^2));
        
        
    end
end

% K_SKIP_6 = U_latlon_6 * diag(D_latlon_6) * U_latlon_6';
% max(max(abs(K_SKIP_6 - K_Beta_6_Full)))


% 

%% Constant terms!!

K_Beta_m1_Const = ones(n, n) * pi_m1^1;

K_Beta_0_Const = ones(n, n) * pi_0^1;

K_Beta_1_LIN = (Mag_vector * Mag_vector') * pi_1^1;

K_Beta_2_LIN = (Mag_vector.^2 * Mag_vector.^2') * pi_2^1;

K_Beta_3_LIN = (log_RJB_vector * log_RJB_vector') * pi_3^1;

%K_Beta_4_LIN = (Mlog_RJB_vector * Mlog_RJB_vector') * pi_4^1;

K_Beta_5_LIN = (RJB_vector * RJB_vector') * pi_5^1;

K_Beta_6_LIN = (VS30_vector * VS30_vector') * pi_6^1;


%% Checks Linear Terms!!

% a = normrnd(0, 1, n, 1);
% 
% b_full = K_Beta_1_LIN * a;
% 
% b_SKIP = Kg1_mvm_function_LIN(a);
% 
% max(abs(b_full - b_SKIP))
%%
% Mixed effects!!


% K_Full_mixed_effects = zeros(n, n);
% 
% for i = 1:n
%     for j = 1:n
%         
%         if Eq_ID_vector(i) == Eq_ID_vector(j)
%             
%             K_Full_mixed_effects(i, j) = 1;
%             
%         end
%         
%     end
%     
% end

%K_Full_mixed_effects = tau^2 * K_Full_mixed_effects;

%% Total Covariance Matrix!

K_Full_Total =  K_Beta_m1_Full + ...
                K_Beta_0_Full + ...
                K_Beta_3_Full + ...
                K_Beta_5_Full + ...
                K_Beta_6_Full + ...
                ...
                K_Beta_m1_Const + ...
                K_Beta_0_Const + ...
                K_Beta_1_LIN + ...
                K_Beta_2_LIN + ...
                K_Beta_3_LIN + ...
                K_Beta_5_LIN + ...
                K_Beta_6_LIN;
                

% K_SKIP_Total = K_mvm_total_SKIP(eye(n, n));
% max(max(abs(K_Full_Total - K_SKIP_Total)))

% tic
alpha_vector_full = (K_Full_Total + sn^2 * eye(n, n))\y;
% toc

% max(max(abs(alpha_vector_full - alpha_vector)))/norm(alpha_vector_full)
% plot(alpha_vector - alpha_vector_full)

time_Full_inf = toc;

%% Covariance matrix for prediction!!


% Beta m1

K_Beta_m1_Full_Pred = zeros(n, n_data_pred);

for i = 1:n
    for j = 1:n_data_pred
        
        K_Beta_m1_Full_Pred(i, j) = sf_m1^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector_Pred(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector_Pred(j))^2)/(2*ell_m1^2));
        
        
    end
end



% K_SKIP_m1_Pred = U_m1_Pred * diag(D_m1_Pred) * V_m1_Pred';
% max(max(abs(K_SKIP_m1_Pred' - K_Beta_m1_Full_Pred)))

% Beta 0

K_Beta_0_Full_Pred = zeros(n, n_data_pred);

for i = 1:n
    for j = 1:n_data_pred
        
        K_Beta_0_Full_Pred(i, j) = sf_0^2 * exp(-((staLatitude_vector(i) - staLatitude_vector_Pred(j))^2 + ...
                                             (staLongitude_vector(i) - staLongitude_vector_Pred(j))^2)/(2*ell_0^2));
        
        
    end
end


% K_SKIP_0_Pred = U_0_Pred * diag(D_0_Pred) * V_0_Pred';
% max(max(abs(K_SKIP_0_Pred' - K_Beta_0_Full_Pred)))


% Beta 3


K_Beta_3_Full_Pred = zeros(n, n_data_pred);

for i = 1:n
    for j = 1:n_data_pred
        
        K_Beta_3_Full_Pred(i, j) = log_RJB_vector(i) * log_RJB_vector(j) * ...
                              sf_3^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector_Pred(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector_Pred(j))^2)/(2*ell_3^2));
        
        
    end
end

% K_SKIP_3_Pred = U_3_Pred * diag(D_3_Pred) * V_3_Pred';
% max(max(abs(K_SKIP_3_Pred' - K_Beta_3_Full_Pred)))




% Beta 5


K_Beta_5_Full_Pred = zeros(n, n_data_pred);

for i = 1:n
    for j = 1:n_data_pred
        
        K_Beta_5_Full_Pred(i, j) = RJB_vector(i) * RJB_vector(j) * ...
                              sf_5^2 * exp(-((eqLatitude_vector(i) - eqLatitude_vector_Pred(j))^2 + ...
                                             (eqLongitude_vector(i) - eqLongitude_vector_Pred(j))^2)/(2*ell_5^2));
        
        
    end
end

% K_SKIP_5_Pred = U_5_Pred * diag(D_5_Pred) * V_5_Pred';
% max(max(abs(K_SKIP_5_Pred' - K_Beta_5_Full_Pred)))


% Beta 6 

K_Beta_6_Full_Pred = zeros(n, n_data_pred);

for i = 1:n
    for j = 1:n_data_pred
        
        K_Beta_6_Full_Pred(i, j) = VS30_vector(i) * VS30_vector(j) * ...
                              sf_6^2 * exp(-((staLatitude_vector(i) - staLatitude_vector_Pred(j))^2 + ...
                                             (staLongitude_vector(i) - staLongitude_vector_Pred(j))^2)/(2*ell_6^2));
        
        
    end
end

% K_SKIP_6_Pred = U_6_Pred * diag(D_6_Pred) * V_6_Pred';
% max(max(abs(K_SKIP_6_Pred' - K_Beta_6_Full_Pred)))


% 

% Constant terms!!

K_Beta_m1_Const_Pred = ones(n, n_data_pred) * pi_m1^1;
K_Beta_0_Const_Pred = ones(n, n_data_pred) * pi_0^1;
K_Beta_1_LIN_Pred = (Mag_vector  * Mag_vector(1:n_data_pred)') * pi_1^1;
K_Beta_2_LIN_Pred = (Mag_vector.^2  * Mag_vector(1:n_data_pred)'.^2) * pi_2^1;
K_Beta_3_LIN_Pred = (log_RJB_vector * log_RJB_vector(1:n_data_pred)') * pi_3^1;
K_Beta_5_LIN_Pred = (RJB_vector * RJB_vector(1:n_data_pred)') * pi_5^1;
K_Beta_6_LIN_Pred = (VS30_vector * VS30_vector(1:n_data_pred)') * pi_6^1;


K_Full_Total_Pred =  K_Beta_m1_Full_Pred + ...
                    K_Beta_0_Full_Pred + ...
                    K_Beta_3_Full_Pred + ...
                    K_Beta_5_Full_Pred + ...
                    K_Beta_6_Full_Pred + ...
                    ...
                    K_Beta_m1_Const_Pred + ...
                    K_Beta_0_Const_Pred + ...
                    K_Beta_1_LIN_Pred + ...
                    K_Beta_2_LIN_Pred + ...
                    K_Beta_3_LIN_Pred + ...
                    K_Beta_5_LIN_Pred + ...
                    K_Beta_6_LIN_Pred;

% K_SKIP_Total_Pred = LHS_Matrix_Pred * RHS_Matrix_Pred';
% max(max(abs(K_Full_Total_Pred - K_SKIP_Total_Pred)))       

%% Median Prediction!!

mu_vector_full = K_Full_Total_Pred' * alpha_vector_full; % (K_{X*, X})* alpha = K_{X*, X} * alpha!! No noise term in K_{X*, X} for predictions!! 

% max(abs(mu_vector_full - mu_vector))

%% Epistemic uncertainty!!

kss_vector_full = sf_m1^2 * ones(n_data_pred, 1) +  ...
                  sf_0^2 * ones(n_data_pred, 1) + ...
                  sf_3^2 * log_RJB_vector(1:n_data_pred, 1).^2 + ...
                  sf_5^2 * RJB_vector(1:n_data_pred, 1).^2 + ...
                  sf_6^2 * VS30_vector(1:n_data_pred, 1).^2 + ...
                  ...
                  pi_m1 * ones(n_data_pred, 1) + ...
                  pi_0 * ones(n_data_pred, 1) + ...
                  pi_1 * Mag_vector(1:n_data_pred, 1).^2 + ...
                  pi_2 * Mag_vector_2(1:n_data_pred, 1).^2 + ...
                  pi_3 * log_RJB_vector(1:n_data_pred, 1).^2 + ...
                  pi_5 * RJB_vector(1:n_data_pred, 1).^2 + ...
                  pi_6 * VS30_vector(1:n_data_pred, 1).^2;
              
              

%%% Cross-covariance diagonal vector K_{X*, X} * K_{X, X}^{-1}*K_{X, X*}!!

temp_Matrix = K_Full_Total_Pred' * ((K_Full_Total + sn^2 * eye(n, n)) \ K_Full_Total_Pred); 

Psi_vector_full = kss_vector_full - diag(temp_Matrix);

% max(max(abs(Psi_vector_full - Psi_vector)))

%% Plots Median and Epistemic Uncertainty from both approaches

fs = 17;
lw = 1;

% Median prediction

figure
plot(mu_vector_full - mu_vector, '--', 'Linewidth', lw)
set(gca, 'Fontsize', fs)

xlabel('Scenario Number')
ylabel('Median relative error')

% Epistemic uncertainty error

figure 
plot(Psi_vector_full - Psi_vector, '--', 'Linewidth', lw)
set(gca, 'Fontsize', fs)

xlabel('Scenario Number')
ylabel('\Psi relative error')

%% Time and Memory checks


time_Full = toc;


[user,sys] = memory;
memory_used = user.MemUsedMATLAB/1E9 - memory_used_initial;

disp(strcat('Time elapsed with Full approach is: ', {' '},  num2str(time_Full)))

disp(strcat('Memory used with Full approach is: ', {' '},  num2str(memory_used), 'GB'))
