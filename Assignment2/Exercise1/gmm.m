% Exercise-1 Zhen Zhou 03721400

X=importdata('dataGMM.mat'); 
X=X';
K=4;
[PX, Model] = Gmm(X, K);

% Plot the density values 
[X1,X2] = meshgrid(linspace(-0.1,0.1,100)', linspace(-0.1,0.1,100)');
X = [X1(:) X2(:)];
for i=1:K
p = mvncdf(X, Model.Miu(i,:), Model.Sigma(:,:,i));
end
surf(X1,X2,reshape(p,100,100)); 

% train Gaussian Mixture model
function varargout = Gmm(X, K)
    threshold = 1e-20; % set small threshold
    [N, D] = size(X); 

    [pMiu, pPi, pSigma] = init_params(); % iInitialize parameters
 
    Lbef = -inf;
    % E-M algorithm
    while true
        Px = calc_prob(); % calculate probability
        
        % new value for pOmega
        pOmega = Px .* repmat(pPi, N, 1);
        pOmega = pOmega ./ repmat(sum(pOmega, 2), 1, K);
 
        % new value for parameters of each Component
        Nk = sum(pOmega, 1);
        pMiu = diag(1./Nk) * pOmega' * X;
        pPi = Nk/N;
        for c = 1:K
            Xdiff = X-repmat(pMiu(c, :), N, 1);
            pSigma(:, :, c) = (Xdiff' * (diag(pOmega(:, c)) * Xdiff)) / Nk(c);
        end
 
        % check for convergence
        L = sum(log(Px*pPi'));  % log-likelihood
        if L-Lbef < threshold
            break;
        end
        Lbef = L;
    end
 
        model = [];
        model.Miu = pMiu;
        model.Sigma = pSigma;
        model.Pi = pPi;
        varargout = {Px, model};

%--------------------------------------------------------------------------
      %% iInitialize parameters(pMiu,pPi,pSigma)
      function [pMiu,pPi,pSigma] = init_params()
          [labels, pMiu] = kmeans(X, K);
          pPi = zeros(1, K);
          pSigma = zeros(D, D, K);
         % For each cluster, evaluate pi and sigma.
          for i=1:K
              Xk = X(labels == i, :);
              pPi(i) = size(Xk, 1) / N;
              pSigma(:, :, i) = cov(Xk);
          end
      end
    
     %% calculate probability
     function Px = calc_prob()
         Px = zeros(N, K);
         for k = 1:K
             Xdiff = X-repmat(pMiu(k, :), N, 1);
             inv_pSigma = inv(pSigma(:, :, k));
             tmp = sum((Xdiff*inv_pSigma).* Xdiff, 2);
             coef = (2*pi)^(-D/2) * sqrt(det(inv_pSigma));
             Px(:, k) = coef * exp(-0.5*tmp);
         end
     end
end

