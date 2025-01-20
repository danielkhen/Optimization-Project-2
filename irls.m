function [x, losses] = irls(x, A, L, y, alpha, epsilon, tol, cglsTol, maxIter, cglsMaxIter)
    % Solves: min_x (1/2 * ||y - Ax||^2 + alpha * ||Lx||_1)
    %
    % Inputs:
    % x - initial solution
    % A - Sparse matrix for projections (size m x n)
    % L - Regularization matrix (size p x n)
    % y - Observations (vector of size m)
    % alpha - Regularization parameter
    % epsilon - tolerance for least squares approximation
    % tol - Convergence tolerance
    % cglsTol - CGLS convergence tolerance
    % maxIter - Maximum number of iterations
    % cglsMaxIter - maximum number of iterations for cgls
    %
    % Output:
    % x - Solution vector (size n)

    loss = 0.5 * norm(y - A * x)^2 + alpha * sum(abs(L * x));
    fprintf('starting objective value: %f\n', loss);
    losses = [];

    for iter = 1:maxIter
        % Compute Lx and update W
        Lx = L * x;
        sqrt_W_diag = 1 ./ sqrt(max(abs(Lx), epsilon)); % Diagonal entries of W^1/2
        sqrt_W = spdiags(sqrt_W_diag, 0, length(sqrt_W_diag), length(sqrt_W_diag)); % Sparse W^1/2
        L_tag = sqrt_W * L;

        % Solve the system using Conjugate Gradient Least Squares
        x_new = cgls(A, L_tag, y, alpha * 2, cglsTol, cglsMaxIter);

        % Check for convergence
        if norm(x_new - x) / norm(x) < tol
            fprintf('IRLS converged after %d iterations\n', iter);
            x = x_new;
            break;
        end

        % Update x for the next iteration
        x = x_new;
        loss = 0.5 * norm(y - A * x)^2 + alpha * sum(abs(L * x));
        losses = [losses, loss];
        fprintf('Iteration %d complete, objective value: %f\n', iter, loss);
    end

    if iter == maxIter
        fprintf('IRLS reached maximum iterations\n');
    end
end
