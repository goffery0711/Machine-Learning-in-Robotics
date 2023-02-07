% Exercise-2 Zhen Zhou 03721400
% Ideal£ºuse forward algorithm to get alpha-value(P1)

%% Initialization
load('Test.txt');
load('A.txt');
load('B.txt');
load('pi.txt');

NC = size(Test, 2); % NC=10:Number of total sequences
N = size(A, 1);     % N = 12:the number of states
gesture = zeros(1, NC); % 1*10:Classification of gesture 1 or 2


%% Iterate from 1.sequence to 10. sequence
for s = 1:NC  % s:sequences
    cs = Test(:, s);   % cs:The current sequence to be operated
    T = size(cs, 1);   % T = 60:the time of states
    alpha = zeros(T, N);  % Alpha matrix represents forward probability matrix
    s_alpha = zeros(N,1); % sum of Alpha matrix
%   P1 = zeros(1,NC);  % as likelihood 
    
    
    % step1. Initialization
    for i = 1:N
        alpha(1, i) = pi(1,i) * B(cs(1,1),i); 
    end
    
    % step2. Recursively get the forward probability 
    %        and stores it in matrix P1
    for t = 1:(T-1)
        for j = 1:N
            for i = 1:N
               s_alpha(i,1) = alpha(t,i) .* A(i, j);
            end
            alpha(t+1,j) = sum(s_alpha) * B(cs(t+1,1),j); 
        end
    end
    P1(s) = sum(alpha(T,:));
    P1(s) = log(P1(s));      % log-likelihood
    
    % step3. Classification
    if P1(s) > -115
       gesture(s) = 1;
    else
       gesture(s) = 2;
    end
end

gesture