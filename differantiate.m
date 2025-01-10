function [DxX, DyX] = differantiate(X)
    [M, N] = size(X);
    X = reshape(X, M*N, 1);
    [Dx, Dy] = differential_matrices(M, N);
    DxX = reshape(Dx * X, M, N);
    DyX = reshape(Dy * X, M, N);
end