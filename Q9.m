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
[Dx, Dy] = differential_matrices(M, N);
L = [Dx; Dy];
Q = A' * A + lambda * (L' * L);

eigenvalues = eig(Q);
condition_number = eigenvalues(25) / eigenvalues(1);
fprintf('Condition number (highest divided by lowest eigenvalues): %d.\n', condition_number);

format long

k = log(1/(10*condition_number)) / log(1 - 1/condition_number);
fprintf('Iteration for lowering the objective by 10: %d.\n', k);