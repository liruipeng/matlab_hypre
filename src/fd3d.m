function A = fd3d(nx,ny,nz,alpx,alpy,alpz,dshift)
% 
% NOTE nx and ny must be > 1 -- nz can be == 1.
%
% 5- or 7-point block-Diffusion/conv. matrix. with
%
tx = tridiag(2, -1+alpx, -1-alpx, nx) ;
ty = tridiag(2, -1+alpy, -1-alpy, ny) ;
tz = tridiag(2, -1+alpz, -1-alpz, nz) ;
A = kron(speye(ny,ny),tx) + kron(ty,speye(nx,nx)); 
if (nz > 1) 
     A = kron(speye(nz,nz),A) + kron(tz,speye(nx*ny,nx*ny)); 
end
A = A - dshift * speye(nx*ny*nz,nx*ny*nz);
