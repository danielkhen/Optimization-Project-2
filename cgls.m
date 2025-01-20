% Function Definition
function x = cgls(A, L, y, lambda, tol, maxIter)
    % Solves: min_x (1/2 * ||y - Ax||^2 + lambda/2 * ||Lx||^2)
    %
    % Inputs:
    % A - Sparse matrix for projections (size m x n)
    % L - Regularization matrix (size p x n)
    % y - Observations (vector of size m)
    % lambda - Regularization parameter
    % tol - Convergence tolerance
    % maxIter - Maximum number of iterations
    %
    % Output:
    % x - Solution vector (size n)
    
    % Initialize variables
    [m, n] = size(A);        % Dimensions of A
    x = zeros(n, 1);         % Initial guess
    r = A' * y;              % Residual
    d = r;                   % Initial direction
    norm_y = norm(y);        % Norm of y for relative tolerance
    LTL = lambda * (L' * L); % Precompute L^T * L (sparse-friendly)
    
    % Main CGLS loop
    for k = 1:maxIter
        % Compute Ad and Ld
        Ad = A * d;
        Ld = L * d;
        
        % Compute alpha
        alpha = (r' * r) / (Ad' * Ad + Ld' * Ld);
        
        % Update x
        x = x + alpha * d;
        
        % Update residual
        r_new = r - alpha * (A' * Ad + LTL * d);
        
        % Check convergence
        if norm(r_new) / norm_y < tol
            fprintf('Converged in %d iterations.\n', k);
            break;
        end
        
        % Update beta
        beta = (r_new' * r_new) / (r' * r);
        
        % Update direction
        d = r_new + beta * d;
        
        % Update residual
        r = r_new;
    end
    
    if k == maxIter
        fprintf('Reached maximum iterations (%d), %d .\n', maxIter, norm(r_new) / norm_y);
    end
end


