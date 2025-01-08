function [Dx, Dy] = differential_matrices(M, N)
    size = M*N;
    indices = 1:size;
    values = [repelem(1, size-M), repelem(-1, size-M)]; % Assign M*N ones then M*N minus ones
    
    % Ones assigned to diagonals and minus ones assigned to diagonal moved
    % right by one but without modulo M indices or last row input indices
    without_last_row = indices(mod(indices, M) ~= 0);
    Dx_rows = [without_last_row, without_last_row]; 
    Dx_cols = [without_last_row, without_last_row + 1];
    
    % Ones assigned to diagonals and minus ones assigned to diagonal moved
    % right by M but without last M indices or last col input indices
    without_last_col = 1:size-M;
    Dy_rows = [without_last_col, without_last_col];
    Dy_cols = [without_last_col, without_last_col + M];

    Dx = sparse(Dx_rows, Dx_cols, values, size, size);
    Dy = sparse(Dy_rows, Dy_cols, values, size, size);
end

function [DxX, DyX] = differantiate(X)
    [M, N] = size(X);
    X = reshape(X, M*N, 1);
    [Dx, Dy] = differential_matrices(M, N);
    DxX = reshape(Dx * X, M, N);
    DyX = reshape(Dy * X, M, N);
end

function Y = normalize(X)
    min_X = min(X,[],"all");
    max_X = max(X,[],"all");
    Y = (X - min_X) / (max_X - min_X);
end

load("X1.mat")
load("X2.mat")
load("X3.mat")

[DxX1, DyX1] = differantiate(X1);
[DxX2, DyX2] = differantiate(X2);
[DxX3, DyX3] = differantiate(X3);

MX1 = sqrt(DxX1.^2 + DyX1.^2);
MX2 = sqrt(DxX2.^2 + DyX2.^2);
MX3 = sqrt(DxX3.^2 + DyX3.^2);

subplot(3, 4, 1);
imshow(normalize(X1));
title('Original')
subplot(3, 4, 2);
imshow(normalize(DxX1));
title('Vertical differetial')
subplot(3, 4, 3);
imshow(normalize(DyX1));
title('Horizontal differetial')
subplot(3, 4, 4);
imshow(normalize(MX1));
title('Differential magnitude')

subplot(3, 4, 5);
imshow(normalize(X2));
title('Original')
subplot(3, 4, 6);
imshow(normalize(DxX2));
title('Vertical differetial')
subplot(3, 4, 7);
imshow(normalize(DyX2));
title('Horizontal differetial')
subplot(3, 4, 8);
imshow(normalize(MX2));
title('Differential magnitude')

subplot(3, 4, 9);
imshow(normalize(X3));
title('Original')
subplot(3, 4, 10);
imshow(normalize(DxX3));
title('Vertical differetial')
subplot(3, 4, 11);
imshow(normalize(DyX3));
title('Horizontal differetial')
subplot(3, 4, 12);
imshow(normalize(MX3));
title('Differential magnitude')