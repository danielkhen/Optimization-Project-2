%% Q3 - Differential matrices
M = 3;
N = 4;

[Dx, Dy] = differential_matrices(M, N);

fprintf("Dx matrix for size %d x %d\n", M, N);
disp(full(Dx));

fprintf("Du matrix for size %d x %d\n", M, N);
disp(full(Dy));