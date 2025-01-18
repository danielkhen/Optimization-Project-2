%% Q10 - Conjugate Gradient Least Squares (CGLS)

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



%%

Y = load("Y.mat");
Y = Y.Y;
y = Y(:); % Flatten Y into a column vector
y = nonzeros(Y);
M = 5;
N = 5;
size = M * N;

lambda = 1e-5;

A_cols = {
    [4, 10];
    [2, 8, 14, 20];
    [16, 17, 18, 19, 20];
    [4, 9, 14, 19, 24];
    [6, 12, 18, 24];
    [2, 7, 12, 17, 22];
    [1, 6, 11, 16, 21];
    [5, 9, 13, 17, 21]
};

observations = length(A_cols);

cols_multiplier = [sqrt(2), sqrt(2), 1, 1, sqrt(2), 1, 1, sqrt(2)];
A_rows = cellfun(@(list, factor) repelem(factor, length(list)), A_cols, num2cell(reshape(1:observations,observations,1)), 'UniformOutput', false);
A_values = cellfun(@(list, factor) repelem(factor, length(list)), A_cols, num2cell(reshape(cols_multiplier,observations,1)), 'UniformOutput', false);

A_cols = horzcat(A_cols{:});
A_rows = horzcat(A_rows{:});
A_values = horzcat(A_values{:});

A = sparse(A_rows, A_cols, A_values, observations, size);

disp(A);

% Construct derivative matrices
[Dx, Dy] = differential_matrices(M, N);
L = [Dx; Dy];

% Parameters for regularization
lambda = 1e-5;
tol = 1e-5;
maxIter = 5000000; % Set high number to ensure convergence

% Solve using CGLS
x = cgls(A, L, y, lambda, tol, maxIter);




%% Q11
% Load data
A = load('Small/A.mat'); % Load ray-path matrix

A = A.A;
Y = load('Small/y.mat'); % Load observations
Y = Y.y;

% Define regularization matrices
n = size(A, 2); % Number of columns in A (6859 here for Small)
Dx = spdiags([-ones(n,1), ones(n,1)], [0, 1], n, n); % Forward diff in x
Dy = spdiags([-ones(n,1), ones(n,1)], [0, 1], n, n); % Forward diff in y
L = [Dx; Dy]; % Regularization matrix (concatenation)

% Set parameters
lambda = 1e-5; % Regularization parameter
tol = 1e-5;    % Convergence tolerance
maxIter = 2000; % Maximum number of iterations

% Solve with CGLS
x = cgls(A, L, Y, lambda, tol, maxIter);

% Reshape to 3D
dim = round(nthroot(length(x), 3)); % Determine 3D grid size
if dim^3 ~= length(x)
    error('Vector length does not match a cubic grid.');
end
X = reshape(x, [dim, dim, dim]); % Reshape to 3D volume

% Use displayVolumeSliceGUI to visualize the 3D volume
displayVolumeSliceGUI(X);


%% Q11 - 3D Reconstruction with CGLS

% Load data
data = load('Small/y.mat'); % Normalized attenuation (observations)
y = data.y;

ray_path_data = load('Small/A.mat'); % Ray path matrix
A = ray_path_data.A;

% Dimensions of the 3D volume (Small bag)
n = 19; % Grid size (19x19x19)
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
lambda_values = [1e-5, 1e-2, 1]; % Test with different lambda values
tol = 1e-5; % Convergence tolerance
maxIter = 20000; % Maximum number of iterations

for lambda = lambda_values
    fprintf('\nSolving with lambda = %.1e\n', lambda);
    
    % Solve using CGLS
    x = cgls(A, L, y, lambda, tol, maxIter);
    
    % Reshape the solution to 3D volume
    X = reshape(x, [n, n, n]);
    
    % Compute and display the objective value
    obj_value = 0.5 * norm(y - A * x)^2 + (lambda / 2) * norm(L * x)^2;
    fprintf('Objective value: %.4f\n', obj_value);

    % Display the reconstructed volume
    displayVolumeSliceGUI(X);

    % Add a title to the figure
    fig = gcf; % Get current figure handle
    sgtitle(sprintf('3D Reconstruction with \\lambda = %.1e', lambda));
end
