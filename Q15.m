%% Q15 - Iteratively reweighted Least Squares (IRLS)

% Function Definition
function x = cgls(A, L, y, lambda, tol, maxIter)
    % Solves: min_x (1/2 * ||y - Ax||^2 + lambda/2 * ||Lx||^2)
    %
    % Inputs:
    % x - Initial guess
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
    x = zeros(n, 1);         % Initial gues
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

function x = irls(x, A, L, y, alpha, epsilon, tol, cglsTol, maxIter, cglsMaxIter)
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

        fprintf('Iteration %d complete, objective value: %f\n', iter, ...
            0.5 * norm(y - A * x)^2 + alpha * sum(abs(L * x)));
    end

    if iter == maxIter
        fprintf('IRLS reached maximum iterations\n');
    end
end

% Load data
data = load('Large/y.mat'); % Normalized attenuation (observations)
y = data.y;

ray_path_data = load('Large/A.mat'); % Ray path matrix
A = ray_path_data.A;

% Dimensions of the 3D volume (Small bag)
n = 49; % Grid size (19x19x19)
size_3D = n^3; % Total voxels in the 3D grid

% Construct 3D finite difference derivative matrices
I = speye(n);
Dx = kron(speye(n^2), spdiags([-ones(n,1), ones(n,1)], [0,1], n, n));
Dy = kron(speye(n), kron(spdiags([-ones(n,1), ones(n,1)], [0,1], n, n), I));
Dz = kron(spdiags([-ones(n,1), ones(n,1)], [0,1], n, n), speye(n^2));

% Apply Neumann boundary conditions (zero gradient at boundaries)
Dx(end-n+1:end, :) = 0;
Dy(end-n^2+1:end, :) = 0;
Dz(end-n^3+1:end, :) = 0;

% Regularization matrix
L = [Dx; Dy; Dz];

% Regularization parameters
alpha = 5e-1;
lambda = 1e-5;
tol = 1e-2; % Convergence tolerance
cglsTol = 1e-5; % CGLS convergence tolerance
cglsMaxIter = 1000; % Maximum number of iterations
maxIter = 10;

x_start = cgls(A, L, y, lambda, cglsTol, cglsMaxIter);
x = irls(x_start, A, L, y, alpha, epsilon, tol, cglsTol, maxIter, cglsMaxIter);

% Reshape the solution to 3D volume
X = reshape(x, [n, n, n]);
X_start = reshape(x_start, [n, n, n]);

% Display the reconstructed volume
displayVolumeSliceGUI(X_start);
fig = gcf; % Get current figure handle
sgtitle(sprintf('L2 regularization'));

% Display the reconstructed volume
displayVolumeSliceGUI(X);
fig = gcf; % Get current figure handle
sgtitle(sprintf('L1 regularization'));

%x_slices = [6, 9, 12];
%subplot(1, 2, 1);
%slice(X_start, x_slices, [], []); % Reconstruction from problem (3)
%title('L2 regularization');

%subplot(1, 2, 2);
%slice(X, x_slices, [], []); % Reconstruction from problem (6)
%title('L1 regularization');