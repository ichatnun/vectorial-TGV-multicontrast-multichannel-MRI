function [rmse, mean_rmse] = computeRMSE3D( x_recon, x_true, mask_rmse )
[N(1),N(2),N(3),L] = size(x_true);
rmse = zeros(N(3),L);
for slice = 1:N(3)
    for i = 1:L 
        mask = mask_rmse(:,:,slice,i);
        true = x_true(:,:,slice,i);
        recon = x_recon(:,:,slice,i);
        rmse(slice,i) = 100*norm(true(mask)-recon(mask))/norm(true(mask));
    end
end
mean_rmse = mean(rmse(:));
