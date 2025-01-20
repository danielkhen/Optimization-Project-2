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
