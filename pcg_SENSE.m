function res = pcg_SENSE(x, sens, M, N, N_ch)

x_reshape = reshape(x,N(1),N(2));

for chan = 1:N_ch
    if chan == 1
        res = conj(sens(:,:,chan)).*ifft2(abs(M).^2.*fft2(sens(:,:,chan).*x_reshape));
    else
        res = res + conj(sens(:,:,chan)).*ifft2(abs(M).^2.*fft2(sens(:,:,chan).*x_reshape));
    end
end

res = res(:);