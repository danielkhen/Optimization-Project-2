%% Q16 - The large bag
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
epsilon = 1e-3;
alpha = 5e-1;
lambda = 1e-5;
tol = 1e-3; % Convergence tolerance
cglsTol = 1e-5; % CGLS convergence tolerance
cglsMaxIter = 1000; % Maximum number of iterations
maxIter = 100;

x_start = cgls(A, L, y, lambda, cglsTol, cglsMaxIter);
[x, losses] = irls(x_start, A, L, y, alpha, epsilon, tol, cglsTol, maxIter, cglsMaxIter);

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