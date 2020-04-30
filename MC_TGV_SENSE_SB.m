function [u, obj_val, rmse, mean_rmse, rmse_solution, mssim]=MC_TGV_SENSE_SB_decoupled(x_init,f, sens, M, lambda, a0, a1, iterMax, mu, x_true, mask_rmse,tol_rmse_solution, tol_obj_val, USE_PRECONDITIONER, resolution, R, USE_ISOTROPIC_TGV)




pcg_tol = 1e-4;
pcg_maxIter = 200;
l2=lambda/2;mu2=mu/2;
[m,n,L, N_ch]=size(f);
N(1) = m; N(2) = n;
u=x_init;
v=zeros(m,n,L,2,'single');
p=v; s=v;
q=zeros(m,n,L,3); b=q;

% Prepare this for ssim
img_true = zeros([N,L]);
for h = 1:L
    img_true(:,:,h) = im256(x_true(:,:,h).*mask_rmse(:,:,h));
end

if USE_PRECONDITIONER
    diag_DFtDFx = 2*ones(N); diag_DFtDFx(1,:) = 1; diag_DFtDFx(end,:) = 1; diag_DFtDFx = diag_DFtDFx/(resolution(1)^2);
    diag_DFtDFy = 2*ones(N); diag_DFtDFy(:,1) = 1; diag_DFtDFy(:,end) = 1; diag_DFtDFy = diag_DFtDFy/(resolution(2)^2);
    diag_DBtDBx = 2*ones(N)/(resolution(1)^2); diag_DBtDBx(end,:) = 0; 
    diag_DBtDBy = 2*ones(N)/(resolution(2)^2); diag_DBtDBy(:,end) = 0;   
    
    P_u = lambda/2*sum(abs(sens.^2),3)/R + mu/2*(diag_DFtDFx+diag_DFtDFy);
    P_vx = mu/2* (ones(N) + diag_DBtDBx + diag_DBtDBy/2);
    P_vy = mu/2* (ones(N) + diag_DBtDBx/2 + diag_DBtDBy);
end

CtFtMtf = zeros([N,L],'single');
for l = 1:L
    for chan = 1:N_ch
        CtFtMtf(:,:,l) = CtFtMtf(:,:,l) + conj(sens(:,:,chan)).*ifft2(M(:,:,l,chan).*f(:,:,l,chan));
    end
end
    
for iter=1:iterMax
    u_prev=u;
    
    %% Updating u & v
    
    % compute rhs
    p_s = p-s;
    q_b = q-b;
    
    for l = 1:L

        rhs = reshape(              -mu2*DxBD(p_s(:,:,l,1))   -mu2*DyBD(p_s(:,:,l,2))   +l2*CtFtMtf(:,:,l), [], 1);
        rhs = cat(1,rhs, reshape(   -mu2*p_s(:,:,l,1)         -mu2*DxFN(q_b(:,:,l,1))   -mu2*DyFN(q_b(:,:,l,3)), [],1)) ;
        rhs = cat(1,rhs, reshape(   -mu2*p_s(:,:,l,2)         -mu2*DyFN(q_b(:,:,l,2))   -mu2*DxFN(q_b(:,:,l,3)), [],1)) ;

        if USE_PRECONDITIONER
            [res,~,~,pcg_iter(l)] = pcg( @(x)pcg_L2part_SB_MC_TGV(x,sens, squeeze(M(:,:,l,:)), m,n,lambda,mu, N_ch), rhs, pcg_tol, pcg_maxIter, @(x)apply_precond_MC_TGV_SENSE(x,P_u, P_vx, P_vy, N), [], cat(1, reshape(u(:,:,l),[],1), reshape(v(:,:,l,:),[],1)));            
        else
            [res,~,~,pcg_iter(l)] = pcg( @(x)pcg_L2part_SB_MC_TGV(x,sens, squeeze(M(:,:,l,:)), m,n,lambda,mu, N_ch), rhs, pcg_tol, pcg_maxIter, [], [], cat(1, reshape(u(:,:,l),[],1), reshape(v(:,:,l,:),[],1)));
        end
        u(:,:,l) = reshape(res(1:m*n),m,n);
        v(:,:,l,:) = reshape(res(m*n+1:end),m,n,2);
        
    end

    %% Update p
    
    % Gradient of u
    grad_u(:,:,:,1)=DxFN(u);
    grad_u(:,:,:,2)=DyFN(u);
    
    % Sym Gradient of v
    sym_grad_v(:,:,:,1)=DxBD(v(:,:,:,1)); 
    sym_grad_v(:,:,:,2)=DyBD(v(:,:,:,2));
    sym_grad_v(:,:,:,3)=(DyBD(v(:,:,:,1))+DxBD(v(:,:,:,2)))/2;
    
    grad_u_v = grad_u - v;
    grad_u_v_s= grad_u_v +s;
    
    if USE_ISOTROPIC_TGV
        P=sqrt(sum(sum(abs(grad_u_v_s).^2,4),3));
        p=bsxfun(@times, max(1-a1./(mu*P),0), grad_u_v_s);
    else
        p = bsxfun(@times, max(1 - (a1/mu)./(sqrt(sum(abs(grad_u_v_s).^2,3))), 0), grad_u_v_s);
    end
    
    %% Update q
    if USE_ISOTROPIC_TGV
        Q=sqrt(sum(abs(sym_grad_v(:,:,:,1)+b(:,:,:,1)).^2+abs(sym_grad_v(:,:,:,2)+b(:,:,:,2)).^2+2*abs(sym_grad_v(:,:,:,3)+b(:,:,:,3)).^2,3));
        q=bsxfun(@times, max(1-a0./(mu*Q),0),sym_grad_v+b);
    else
        sym_grad_v_b = sym_grad_v + b;
        Q=(sqrt(sum(abs(sym_grad_v_b).^2,3)));
%         Q(:,:,3) = 2*Q(:,:,3);
        q = bsxfun(@times, max(1 - (a0/mu)./Q, 0), sym_grad_v_b);
    end
    
    %% Update s & b
    s=s+(grad_u_v-p);
    b=b+(sym_grad_v-q);
    
    %% Stopping criteria
    rmse_solution(iter)= 100*norm(u(:)-u_prev(:))/norm(u_prev(:));
    obj_val(iter)= MC_TGV_SENSE_Energy(u,v(:,:,:,1),v(:,:,:,2),f,sens,M,N,L,N_ch,a1,a0,lambda); 
    
    if iter > 1
        rmse_obj_val(iter) = 100*abs(obj_val(iter)-obj_val(iter-1))/obj_val(iter-1);
    end
    
    [rmse(:,iter), mean_rmse(iter)] = computeRMSE3D( u, x_true, mask_rmse );
    
    for l =1:L
        img_recon(:,:,l) = im256(u(:,:,l).*mask_rmse(:,:,l));
        mssim(l,iter) = ssim(abs(img_recon(:,:,l)),abs(img_true(:,:,l)));
    end
    
    
    if (iter > 1) && (rmse_solution(iter) < tol_rmse_solution) && ( rmse_obj_val(iter) < tol_obj_val) && (sum(pcg_iter) > 0)
%         disp(['Stopping criteria met with SB iter = ', num2str(iter)])
        break
    end

end

end

function [res]=pcg_L2part_SB_MC_TGV(x,sens, M, m,n,lambda,mu, N_ch)
l2=lambda/2;mu2=mu/2;

u=reshape(x(1:m*n),m,n);
v=reshape(x(m*n+1:end), m,n,2);


gradxF_u=DxFN(u);gradyF_u=DyFN(u);
gradxB_vx=DxBD(v(:,:,1)); gradyB_vy=DyBD(v(:,:,2));
gradxB_vy=DxBD(v(:,:,2)); gradyB_vx=DyBD(v(:,:,1));

CtFtMtMFCu = zeros([m,n],'single');

for chan = 1:N_ch
    CtFtMtMFCu = CtFtMtMFCu + conj(sens(:,:,chan)).*ifft2(M(:,:,chan).*fft2(sens(:,:,chan).*u));
end
res= reshape(l2*CtFtMtMFCu-mu2*DxBD(gradxF_u)-mu2*DyBD(gradyF_u)        +mu2*gradxB_vx                                               +mu2*gradyB_vy, [], 1);
res= cat(1, res, reshape(-mu2*gradxF_u                                  +mu2*v(:,:,1)-mu2*DxFN(gradxB_vx)-(mu/4)*DyFN(gradyB_vx)     -(mu/4)*DyFN(gradxB_vy) ,[],1));
res= cat(1, res, reshape(-mu2*gradyF_u                                  -(mu/4)*DxFN(gradyB_vx)                                      +mu2*v(:,:,2)-mu2*DyFN(gradyB_vy)-(mu/4)*DxFN(gradxB_vy) ,[],1));

end

function [ DxFN_I ] = DxFN( I, hx )
% DxFN Derivada en la primera coordenada discretizada mediant % a0 and a1...order not the same as other parts of the codee diferencias
% adelantadas y condiciones de contorno Neumann
%   Si no hay parametro hx se toma como 1

if (~exist('hx', 'var'))
    hx=1;
end

DxFN_I=zeros(size(I));
DxFN_I(1:end-1, :, :)=diff(I, 1, 1)./hx;

end

function [ DyFN_I ] = DyFN( I, hy )
% DyFN Derivada en la segunda coordenada discretizada mediante diferencias
% adelantadas y condiciones de contorno Neumann
%   Si no hay parametro hx se toma como 1

if (~exist('hy', 'var'))
    hy=1;
end

DyFN_I=zeros(size(I));
DyFN_I(:, 1:end-1, :)=diff(I, 1, 2)./hy;

end

function [ DxBD_I ] = DxBD( I, hx )
% DxBD Derivada en la primera coordenada discretizada mediante diferencias
% atrasadas y condiciones de contorno Dirichlet
%   Si no hay parametro hx se toma como 1

if (~exist('hx', 'var'))
    hx=1;
end

DxBD_I=I;
DxBD_I(2:end-1,:,:)=diff(I(1:end-1,:, :), 1, 1)./hx;
DxBD_I(end, :, :)=-I(end-1,:, :);

end

function [ DyBD_I ] = DyBD( I, hy )
% DyBD Derivada en la segunda coordenada discretizada mediante diferencias
% atrasadas y condiciones de contorno Dirichlet
%   Si no hay parametro hy se toma como 1

if (~exist('hy', 'var'))
    hy=1;
end

DyBD_I=I;
DyBD_I(:,2:end-1, :)=diff(I(:,1:end-1, :), 1, 2)./hy;
DyBD_I(:, end, :)=-I(:, end-1, :);

end

function [en_TGV]=MC_TGV_SENSE_Energy(u,v1,v2,f,sens,M,N,L,N_ch,a1,a0,lambda)

grad_x=DxFN(u);
grad_y=DyFN(u);
mod_gradu_v=sqrt(sum(abs(grad_x-v1).^2+abs(grad_y-v2).^2,3));

sym_gradx=DxBD(v1); 
sym_grady=DyBD(v2);
sym_gradz=(DyBD(v1)+DxBD(v2))/2;
mod_symgradv=sqrt(sum(abs(sym_gradx).^2+abs(sym_grady).^2+2*abs(sym_gradz).^2,3));

% Data consistency
MFCul = zeros([N,L,N_ch]);
for l=1:L
    for chan = 1:N_ch
        MFCul(:,:,l,chan) = M(:,:,l,chan).*fft2(sens(:,:,chan).*u(:,:,l));
    end
end
fidelity= norm(MFCul(:)-f(:))^2;

% Regularization
inta1=sum(mod_gradu_v(:));
inta0=sum(mod_symgradv(:));

en_TGV=a1*inta1     +a0*inta0    +(lambda/2)*fidelity;

end

function res = apply_precond_MC_TGV_SENSE(x,P_u, P_vx, P_vy, N)    
    prod_N = prod(N);
    res = zeros(3*prod_N,1);
    res(1:prod_N) = reshape((1./P_u).*reshape(x(1:prod_N),N),[],1);
    res(prod_N+1:2*prod_N) = reshape((1./P_vx).*reshape(x(prod_N+1:2*prod_N),N),[],1);
    res(2*prod_N+1:end) = reshape((1./P_vy).*reshape(x(2*prod_N+1:end),N),[],1);
end
