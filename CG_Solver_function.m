function [alpha_vector, iter] = CG_Solver_function(A, b, tol, maxit_number)
%% Uses pre-conditioned conjugate gradient method to solve for K * Alpha!!

% Might get pre-conditioner (modified incomplete Cholesky preconditioner) 


maxit = min(size(b,1), maxit_number); % Was:  maxit = min(size(b,1),20);


x0 = zeros(size(b)); x = x0;
if isnumeric(A), r = b-A*x; else r = b-A(x); end, r2 = sum(r.*r,1); r2new = r2;

nb = sqrt(sum(b.*b,1)); flag = 0; iter = 1;
relres = sqrt(r2)./nb; todo = relres>=tol; 

if ~any(todo), flag = 1; alpha_vector = x0; return, end

on = ones(size(b,1),1); r = r(:,todo); d = r;


for iter = 2:maxit
  z = A(d); 
  a = r2(todo)./sum(d.*z,1);
  a = on*a;
  x(:,todo) = x(:,todo) + a.*d;
  r = r - a.*z;
  r2new(todo) = sum(r.*r,1);
  relres = sqrt(r2new)./nb; cnv = relres(todo)<tol; todo = relres>=tol;
  d = d(:,~cnv); r = r(:,~cnv);                           % get rid of converged
  if ~any(todo), flag = 1; break, end
  b = r2new./r2;                                               % Fletcher-Reeves
  d = r + (on*b(todo)).*d;
  r2 = r2new;
end



alpha_vector = x;

end