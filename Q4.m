%% Q4 - Differentiating images
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