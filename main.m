function [Dx, Dy] = derivative_matrices(M, N)
    Dx = zeros(M*N, M*N);
    Dy = zeros(M*N, M*N);
    
    for j = 1:N
        for i = 1:M-1
            idx = i + (j-1) * M;
            Dx(idx, idx) = 1;
            Dx(idx, idx + 1) = -1;
        end
    end
    
    for j = 1:N-1
        for i = 1:M
            idx = i + (j-1) * M;
            Dy(idx, idx) = 1;
            Dy(idx, idx + M) = -1;
        end
    end
end

M = 5;
N = 5;

[Dx, Dy] = derivative_matrices(M, N);
disp('Dx:');
disp(full(Dx));  % Display Dx in full matrix form

disp('Dy:');
disp(full(Dy));  % Display Dy in full matrix form