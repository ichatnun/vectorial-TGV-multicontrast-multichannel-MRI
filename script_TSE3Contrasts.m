%% 2D Reconstruction
clear, clc, close all

USE_ISOTROPIC = 0;          % Use isotropic version of MC-TV or MC-TGV operator
USE_PRECONDITIONER = 1;     % Use preconditioner or not
tol_rmse_solution = 5e-2;   % Stopping criteria: RMSE change between consecutive iterations
tol_obj_val = 1e-3;         % Stopping criteria: Objective value change between consecutive iterations
maxSbIter = 150;            % Maximum number of iterations

load([pwd,'/Data/data_TSE3Contrasts.mat'])
[N(1),N(2),L,N_ch] = size(img);

%% Undersample kspace

% Fully sampled kspace
kspace = zeros([N,L,N_ch], 'single'); 
for c = 1:N_ch
    for h = 1:L
        kspace(:,:,h,c) = fft2(img(:,:,h,c)); 
    end 
end

% Undersampling masks
load([pwd,'/Data/R5_TSE_1D_var_den.mat']); 
m2d = repmat(m2d,[1,1,1,N_ch]); R = numel(m2d)/sum(m2d(:));
kspace_us = kspace.*m2d;

im_zf = zeros([N,L,N_ch], 'single'); 
for c = 1:N_ch
    for h = 1:L
        im_zf(:,:,h,c) = ifft2(kspace_us(:,:,h,c));
    end 
end

y_SB = kspace_us; M = m2d(:,:,:,1);

%%
disp('*********************Begin: SENSE (R=1)************************')
pcgTol = 1e-4;
pcgMaxIter = 300;

y_SB_full = kspace;
M_full = ones(size(M));
x_SENSE = zeros([N,L]);
CtFtMty = zeros([N,L]);
for i = 1:L
    for chan = 1:N_ch
        if chan == 1
            CtFtMty(:,:,i) = conj(sens(:,:,chan)).*ifft2(conj(M_full(:,:,i)).*y_SB_full(:,:,i,chan));
        else
            CtFtMty(:,:,i) = CtFtMty(:,:,i) + conj(sens(:,:,chan)).*ifft2(conj(M_full(:,:,i)).*y_SB_full(:,:,i,chan));
        end
    end
end

for i = 1:L
    b = CtFtMty(:,:,i);
    [temp,pcg_flag,pcg_relres,pcg_iter] = pcg(@(x)pcg_SENSE(x, sens, M_full(:,:,i), N, N_ch) ,b(:),pcgTol,pcgMaxIter,[],[]);
    x_SENSE(:,:,i) = reshape(temp,N);
end
disp('*********************End: SENSE (R=1)************************')
x_true = x_SENSE;

x_zf = squeeze(bsxfun(@rdivide, sum(bsxfun(@times, im_zf , permute(conj(sens),[1,2,4,3])),4) , sum(abs(sens).^2,3) + eps)).*mask_rmse;

%% MC-TGV-SENSE
disp('*********************Begin: MC-TGV-SENSE (SB)************************')
M = single(m2d);                    % Undersampling mask                    
y_TGV= y_SB;                        % Observed k-space data
x_init = x_zf;                      % Initial solution

mu = 1e-1;                          % Augmented Lagrangian parameter

% The following four variables are used to generate the regularization
% parameters
scale_tgv = 3.5;   
a1 = 0.001/2*scale_tgv;             % "1st order"
a0 = 0.001*scale_tgv;               % "2nd order"
lambda = 5;
    
tic
[x_MC_TGV_SENSE,obj_val_MC_TGV_SENSE_iter, rmse_MC_TGV_SENSE_iter, mean_rmse_MC_TGV_SENSE_iter, rmse_solution_MC_TGV_SENSE_iter, mssim_MC_TGV_SENSE_iter]=MC_TGV_SENSE_SB(x_init,y_TGV, sens, M, lambda, a0, a1, maxSbIter, mu, x_true, mask_rmse,tol_rmse_solution, tol_obj_val, USE_PRECONDITIONER, resolution, R,USE_ISOTROPIC);            
t_MC_TGV_SENSE=toc;

[rmse_MC_TGV_SENSE, mean_rmse_MC_TGV_SENSE] = computeRMSE3D( x_MC_TGV_SENSE, x_true, mask_rmse );
mssim_display = 0;
for h = 1:L
    mssim_display = mssim_display + mssim_MC_TGV_SENSE_iter(h,end)/L;
end

disp(['RMSE = ',num2str(rmse_MC_TGV_SENSE.'), ',a0 = ', num2str(a0), ', a1 = ', num2str(a1),', lambda = ', num2str(lambda),',mu = ',num2str(mu), ', ssim: ',num2str(mssim_display), ', mean rmse = ',num2str(mean_rmse_MC_TGV_SENSE)])
mosaic(x_MC_TGV_SENSE,1,3,1,'MC-TGV-SENSE',[0,0.35])
disp('*********************Done: MC-TGV-SENSE (SB)************************')


%% MC-TV-SENSE
disp('*********************Begin: MC-TV-SENSE (SB)************************')
G = TVOP2D;                             % Gradient operator

% Split Bregman params & Initialization
lambda = 2e-4;                          % Regularization parameter
mu = 5e-3;                              % Augmented Lagrangian parameter
M = m2d;                                % Undersampling mask
x_init = x_zf;                          % Initial solution

% Iterative solver
tic
[x_MC_TV_SENSE, rmse_MC_TV_SENSE_iter, mean_rmse_MC_TV_SENSE_iter, rmse_solution_MC_TV_SENSE_iter, mssim_MC_TV_SENSE_iter, obj_val_MC_TV_SENSE] = MC_TV_SENSE_SB(x_init, y_SB, sens, G, M, N, L, lambda, mu, maxSbIter, x_true, mask_rmse,tol_rmse_solution, tol_obj_val, USE_PRECONDITIONER, resolution, R, USE_ISOTROPIC);
t_MC_TV_SENSE = toc;

[rmse_MC_TV_SENSE, mean_rmse_MC_TV_SENSE] = computeRMSE3D( x_MC_TV_SENSE, x_true, mask_rmse );

disp(['RMSE = ',num2str(rmse_MC_TV_SENSE.'), ', lambda = ', num2str(lambda),', mu = ', num2str(mu),', ssim: ',num2str(mean(mssim_MC_TV_SENSE_iter(:,end))),', mean rmse = ', num2str(mean_rmse_MC_TV_SENSE)])
mosaic(x_MC_TV_SENSE,1,3,2,'MC-TV-SENSE',[0,0.35])

disp('*********************Done: MC-TV-SENSE (SB)************************')
