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

M = 5;
N = 5;

[Dx, Dy] = differential_matrices(M, N);
disp('Dx:');
disp(full(Dx));  % Display Dx in full matrix form

disp('Dy:');
disp(full(Dy));  % Display Dy in full matrix form