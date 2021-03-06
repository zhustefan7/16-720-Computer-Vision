function fit = affine_ic_nt(img, tmplt, p_init, n_iters, verbose)
% AFFINE_IC_NT - Affine image alignment using IC Newton algorithm
%   FIT = AFFINE_IC_NT(IMG, TMPLT, P_INIT, N_ITERS, VERBOSE)
%   Align the template image TMPLT to an example image IMG using an
%   affine warp initialised using P_INIT. Iterate for N_ITERS iterations.
%   To display the fit graphically set VERBOSE non-zero.
%
%   p_init = [p1, p3, p5
%             p2, p4, p6];
%
%   This assumes greyscale images and rectangular templates.
%
%   c.f. Baker-Matthews

% Iain Matthews, Simon Baker, Carnegie Mellon University, Pittsburgh
% $Id: affine_ic_nt.m,v 1.1.1.1 2003/08/20 03:07:35 iainm Exp $

if nargin<5, verbose = 0; end
if nargin<4, error('Not enough input arguments'); end

% Common initialisation
init_a;

% Pre-computable things ---------------------------------------------------

% 3) Evaluate gradient of T + second derivatives
[nabla_Tx nabla_Ty] = gradient(tmplt);
[d2T_dxdx d2T_dxdy] = gradient(nabla_Tx);
[d2T_dydx d2T_dydy] = gradient(nabla_Ty);

% 4) Evaluate Jacobian - constant for affine warps
dW_dp = jacobian_a(w, h);

% 5) Compute steepest descent images, VT_dW_dp
VT_dW_dp = sd_images(dW_dp, nabla_Tx, nabla_Ty, N_p, h, w);

% 5b) Compute dW_dp' * d2T_dx2 * d_W_p + (nabla_T * d2W_dp2)
H_extra = zeros(N_p * h, N_p * w);
for p=0:N_p - 1
	tx = dW_dp(1:h,(p*w)+1:(p*w)+w);
	ty = dW_dp(h+1:end,(p*w)+1:(p*w)+w);	
	tx2 = (tx .* d2T_dxdx) + (ty .* d2T_dxdy);
	ty2 = (tx .* d2T_dxdy) + (ty .* d2T_dydy);
	
	for q=0:N_p - 1
		qx = dW_dp(1:h,(q*w)+1:(q*w)+w);
		qy = dW_dp(h+1:end,(q*w)+1:(q*w)+w);	
		H_extra((p*h)+1:(p*h)+h,(q*w)+1:(q*w)+w) = (tx2 .* qx) + (ty2 .* qy);
	end
end
	
% 6) Pre-compute part of Hessian
H = hessian(VT_dW_dp, N_p, w);

% Baker-Matthews, Inverse Compositional Algorithm -------------------------

for f=1:n_iters
	% 1) Compute warped image with current parameters
	IWxp = warp_a(img, warp_p, tmplt_pts);

	% 2) Compute error image - NB reversed
	error_img = IWxp - tmplt;
	
	% -- Save current fit parameters --
	fit(f).warp_p = warp_p;
	fit(f).rms_error = sqrt(mean(error_img(:) .^2));
	
	% -- Show fitting? --
	if verbose
		disp(['Inverse-Compositional NT [',num2str(f-1),']: RMS = ',num2str(fit(f).rms_error)]);
		verb_plot_a(verb_info, warp_p, tmplt_pts, error_img);
	end
	
	% -- Really iteration 1 is the zeroth, ignore final computation --
	if (f == n_iters) break; end
	
	% 6b) Compute Hessian and inverse
	H_nt = zeros(N_p, N_p);
	for p=0:N_p - 1
		for q=0:N_p - 1
			H_nt(p+1,q+1) = sum(sum(H_extra((p*h)+1:(p*h)+h,(q*w)+1:(q*w)+w) .* error_img));
		end
	end
	H_nt = H_nt + H;
	H_inv = inv(H_nt);

	% 7) Compute steepest descent parameter updates
	dG_dp = sd_update(VT_dW_dp, error_img, N_p, w);

	% 8) Compute gradient descent parameter updates
	delta_p = H_inv * dG_dp;
		
	% 9) Update warp parmaters
	warp_p = update_step(warp_p, delta_p);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function warp_p = update_step(warp_p, delta_p)
% Compute and apply the update

delta_p = reshape(delta_p, 2, 3);
	
% Convert affine notation into usual Matrix form - NB transposed
delta_M = [delta_p; 0 0 1];	
delta_M(1,1) = delta_M(1,1) + 1;
delta_M(2,2) = delta_M(2,2) + 1;

% Invert compositional warp
delta_M = inv(delta_M);

% Current warp
warp_M = [warp_p; 0 0 1];	
warp_M(1,1) = warp_M(1,1) + 1;
warp_M(2,2) = warp_M(2,2) + 1;

% Compose
comp_M = warp_M * delta_M;	
warp_p = comp_M(1:2,:);
warp_p(1,1) = warp_p(1,1) - 1;
warp_p(2,2) = warp_p(2,2) - 1;
