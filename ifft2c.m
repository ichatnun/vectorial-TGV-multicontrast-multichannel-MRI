function x = ifft2c(X)

x = ifftshift(ifft2(ifftshift(X)));