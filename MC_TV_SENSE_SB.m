%% Minimize (over X) ||FX-Y||_Frobenius + lambda ||GX||_(2,1) using SB
% x_init                : Initial solution          (numRow x numCol x numContrast)
% y                     : Observation in k-space    (numRow x numCol x numContrast)
% M                     : Undersampling mask        (numRow x numCol x numContrast)
% N                     : Image size                (numRow x numCol)
% L                     : Number of contrasts
% lambda                : Regularization parameter for L21 term
% mu                    : Regularization parameter for the slack-consistency term
% maxSBIter             : Maximum number of SB iterations
% x_true                : True solution (used to compute RMSE. Not available in real life)
% mask_rmse             : Mask (used to compute RMSE)
% tol_RMSE_solution     : tolerance for %rmse change in solution
% tol_obj_func          : Tolerance for the %rmse change in objective function
% USE_PRECONDITIONER    : 1 to apply Jacobi preconditioner for the L2-subproblem
% resolution            : Resolution of the image (used to create Jacobi preconditioner)
% R                     : Reduction factor
% USE_ISOTROPIC_TV      : 1 to use isotropic TV

function [x, rmse, mean_rmse, rmse_solution, mssim, obj_val] = MC_TV_SENSE_SB(x_init, y, sens, G, M, N, L, lambda, mu, maxSbIter, x_true, mask_rmse,tol_rmse_solution, tol_obj_func, USE_PRECONDITIONER, resolution, R, USE_ISOTROPIC_TV)

    pcgTol = 1e-4;
    pcgMaxIter = 200;

    if length(N) == 3

        img_true = zeros([N,L]);
        for slice = 1:N(3)
            for h = 1:L
                img_true(:,:,slice,h) = im256(x_true(:,:,slice,h).*mask_rmse(:,:,slice,h));
            end
        end

        N_ch = size(sens,4);
        CtFtMty = zeros([N,L]);

        % Preconditioner
        if USE_PRECONDITIONER
            P = sum(abs(sens.^2),4)/R + mu*2*sum(reshape(1./resolution.^2,[],1));
        end

        % Precomputation
        for i = 1:L
            for chan = 1:N_ch
                if chan == 1
                    CtFtMty(:,:,:,i) = conj(sens(:,:,:,chan)).*ifftn(conj(M(:,:,:,i)).*y(:,:,:,i,chan));
                else
                    CtFtMty(:,:,:,i) = CtFtMty(:,:,:,i) + conj(sens(:,:,:,chan)).*ifftn(conj(M(:,:,:,i)).*y(:,:,:,i,chan));
                end
            end
        end

        % Step 0
        x = x_init; s = zeros([N,L,3]); Gx =  zeros([N,L,3]); % Data dimensions = Nx, Ny, Nz, contrast, #gradient directions
        for i = 1:L
            Gx(:,:,:,i,:) = G*x(:,:,:,i);
        end
        x_prev = x_init;
        for sbIter = 1:maxSbIter

            % Step 1: soft-thresholding
            Gx_plus_s = Gx + s;

            if USE_ISOTROPIC_TV
                Z=sqrt(sum(sum(abs(Gx_plus_s).^2,5),4));
                z = bsxfun(@times, max(1 - (2*lambda./(mu*Z)),0),Gx_plus_s);
            else
                z = bsxfun(@times, max(1 - (2*lambda/mu)./(sqrt(sum(abs(Gx_plus_s).^2,4))+eps), 0), Gx_plus_s);
            end

            % Step 2: Update each contrast separately
            z_minus_s = z - s;
            for i = 1:L
                b = CtFtMty(:,:,:,i) + mu*(G'*squeeze(z_minus_s(:,:,:,i,:)));

                if USE_PRECONDITIONER
                    [temp,~,~,pcg_iter(i)] = pcg(@(x)pcg_MC_TV_SENSE_L2Part(x, G, sens, M(:,:,:,i), N, N_ch, mu) ,b(:), pcgTol, pcgMaxIter,@(x)apply_precond_MC_TV_SENSE(x,P,N),[],reshape(x(:,:,:,i),[],1));
                else
                    [temp,~,~,pcg_iter(i)] = pcg(@(x)pcg_MC_TV_SENSE_L2Part(x, G, sens, M(:,:,:,i), N, N_ch, mu) ,b(:), pcgTol, pcgMaxIter,[],[],reshape(x(:,:,:,i),[],1));
                end
                x(:,:,:,i) = reshape(temp,N);
            end

            % Step 3: Update s
            for i = 1:L
                Gx(:,:,:,i,:) = G*x(:,:,:,i);
            end
            s = s - (z-Gx);

            % RMSE and Change in solution
            [rmse(:,:,sbIter), mean_rmse(sbIter)] = computeRMSE3D( x, x_true, mask_rmse );
            rmse_solution(sbIter) = 100*norm(x(:) - x_prev(:))/norm(x_prev(:));

            % SSIM
            for slice = 1:N(3)
                for h = 1:L
                    img_recon = im256(x(:,:,slice,h).*mask_rmse(:,:,slice,h));
                    mssim(slice, h,sbIter) = ssim(abs(img_recon),abs(img_true(:,:,slice,h)));
                end
            end

            % Compute objective function value for the SB formulation
            L2_val = 0;
            for chan = 1:N_ch
                for h = 1:L
                    MFCx_minus_y = M(:,:,:,h).*fftn(sens(:,:,:,chan).*x(:,:,:,h)) - y(:,:,:,h,chan);
                    L2_val = L2_val + norm(MFCx_minus_y(:),2)^2;
                end
            end
            Frob_val = norm(reshape(z-Gx-s,[],1),2)^2;
            L21_val = squeeze(sum(abs(z).^2,4)).^(1/2); L21_val = norm(L21_val(:),1);
            obj_val(sbIter) = 1/4*L2_val + lambda*L21_val + mu/4*Frob_val; 

            if sbIter > 1
                rmse_obj_val(sbIter) = 100*abs(obj_val(sbIter) - obj_val(sbIter-1))/abs(obj_val(sbIter-1));
            end

            if (sbIter > 1) && (rmse_solution(sbIter) < tol_rmse_solution) && (rmse_obj_val(sbIter) < tol_obj_func) && (sum(pcg_iter) > 0)
                break
            end

            x_prev = x;

        end
    elseif length(N) == 2
        img_true = zeros([N,L]);
        for h = 1:L
            img_true(:,:,h) = im256(x_true(:,:,h).*mask_rmse(:,:,h));
        end

        N_ch = size(sens,3);
        CtFtMty = zeros([N,L]);

        for i = 1:L
            for chan = 1:N_ch
                if chan == 1
                    CtFtMty(:,:,i) = conj(sens(:,:,chan)).*ifft2(conj(M(:,:,i)).*y(:,:,i,chan));
                else
                    CtFtMty(:,:,i) = CtFtMty(:,:,i) + conj(sens(:,:,chan)).*ifft2(conj(M(:,:,i)).*y(:,:,i,chan));
                end
            end
        end

        if USE_PRECONDITIONER
            P = sum(abs(sens.^2),3)/R + mu*2*sum(reshape(1./resolution.^2,[],1));
        end

        % Step 0
        x = x_init; s = zeros([N,L,2]); Gx =  zeros([N,L,2]); % Data dimensions = Nx, Ny, contrast, #gradient directions
        for i = 1:L
            Gx(:,:,i,:) = G*x(:,:,i);
        end
        x_prev = x_init;
        for sbIter = 1:maxSbIter

            % Step 1: soft-thresholding
            Gx_plus_s = Gx + s;

            if USE_ISOTROPIC_TV
                Z=sqrt(sum(sum(abs(Gx_plus_s).^2,4),3));
                z = bsxfun(@times, max(1 - (2*lambda./(mu*Z)),0),Gx_plus_s);   
            else
                z = bsxfun(@times, max(1 - (2*lambda/mu)./(sqrt(sum(abs(Gx_plus_s).^2,3))+eps), 0), Gx_plus_s);
            end

            % Step 2: Update each contrast separately
            z_minus_s = z - s;
            for i = 1:L
                b = CtFtMty(:,:,i) + mu*(G'*squeeze(z_minus_s(:,:,i,:)));
                if USE_PRECONDITIONER
                    [temp,~,~,pcg_iter(i)] = pcg(@(x)pcg_MC_TV_SENSE_L2Part(x, G, sens, M(:,:,i), N, N_ch, mu) ,b(:), pcgTol, pcgMaxIter,@(x)apply_precond_MC_TV_SENSE(x,P,N),[],reshape(x(:,:,i),[],1));
                else
                    [temp,~,~,pcg_iter(i)] = pcg(@(x)pcg_MC_TV_SENSE_L2Part(x, G, sens, M(:,:,i), N, N_ch, mu) ,b(:), pcgTol, pcgMaxIter,[],[],reshape(x(:,:,i),[],1));            
                end
                x(:,:,i) = reshape(temp,N);
            end

            % Step 3: Update s
            for i = 1:L
                Gx(:,:,i,:) = G*x(:,:,i);
            end
            s = s - (z-Gx);

            % RMSE and Change in solution
            [rmse(:,sbIter), mean_rmse(sbIter)] = computeRMSE3D( x, x_true, mask_rmse );
            rmse_solution(sbIter) = 100*norm(x(:) - x_prev(:))/norm(x_prev(:));

            % SSIM
            for h = 1:L
                img_recon = im256(x(:,:,h).*mask_rmse(:,:,h));
                mssim(h,sbIter) = ssim(abs(img_recon),abs(img_true(:,:,h)));
            end

            % Compute objective function value for the SB formulation
            L2_val = 0;
            for chan = 1:N_ch
                for h = 1:L
                    MFCx_minus_y = M(:,:,h).*fft2(sens(:,:,chan).*x(:,:,h)) - y(:,:,h,chan);
                    L2_val = L2_val + norm(MFCx_minus_y(:),2)^2;
                end
            end
            L21_val = squeeze(sum(abs(Gx).^2,3)).^(1/2); L21_val = norm(L21_val(:),1);
            obj_val(sbIter) = 1/4*L2_val + lambda*L21_val; 

            if sbIter > 1
                rmse_obj_val(sbIter) = 100*abs(obj_val(sbIter) - obj_val(sbIter-1))/abs(obj_val(sbIter-1));
            end

            if (sbIter > 1) && (rmse_solution(sbIter) < tol_rmse_solution) && (rmse_obj_val(sbIter) < tol_obj_func) && (sum(pcg_iter)>0)
                    break
            end

            x_prev = x;

        end
    end

end


function res = pcg_MC_TV_SENSE_L2Part(x, G, sens, M, N, N_ch, mu)

    x_reshape = reshape(x,N);

    if length(N) == 2
        for chan_idx = 1:N_ch
            if chan_idx == 1
                res = conj(sens(:,:,chan_idx)).*ifft2(abs(M).^2.*fft2(sens(:,:,chan_idx).*x_reshape));
            else
                res = res + conj(sens(:,:,chan_idx)).*ifft2(abs(M).^2.*fft2(sens(:,:,chan_idx).*x_reshape));
            end
        end
    elseif length(N) == 3
        for chan_idx = 1:N_ch
            if chan_idx == 1
                res = conj(sens(:,:,:,chan_idx)).*ifftn(abs(M).^2.*fftn(sens(:,:,:,chan_idx).*x_reshape));
            else
                res = res + conj(sens(:,:,:,chan_idx)).*ifftn(abs(M).^2.*fftn(sens(:,:,:,chan_idx).*x_reshape));
            end
        end
    end

    res = res + mu*(G'*(G*x_reshape));

    res = res(:);
end


function res = apply_precond_MC_TV_SENSE(x,P, N)

    res = reshape((1./P).*reshape(x,N),[],1);
    
end