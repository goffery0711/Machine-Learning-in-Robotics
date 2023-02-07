% Exercise-3 Zhen Zhou 03721400
% Task1 Defining reward function
%&Task2 Applying policy iteration

function [ ] = WalkPolicyIteration(Initial_state)
%% Judge the initial state right or wrong
if Initial_state~=round(Initial_state) || Initial_state < 1 ... 
   || Initial_state > 16
    disp('Input Error! Please enter the correct state£¨range:1-16 and an integer).');
    return;
end

%% Task 1: Define reward function
reward_def = zeros(16, 4);

reward_def(1, 2) = -10; % state 1, right leg back/forward will be penalized
reward_def(1, 4) = -10; %   ...  , left leg    ...

reward_def(2, 3) = -10; % state 2, left leg up/down will be penalized
reward_def(2, 4) = -10; %   ...  ,    ...   back/forward   ...

reward_def(3, 3) = -10; % state 3, left leg up/down will be penalized
reward_def(3, 4) = -10; %   ...  ,    ...   back/forward   ...

reward_def(4, 2) = -10; % state 4, right leg back/forward will be penalized
reward_def(4, 4) = -10; %   ...  , left leg    ...

reward_def(5, 1) = -10; % state 5, right leg up/down will be penalized
reward_def(5, 2) = -10; %   ...  ,  ...  back/forward  ...

reward_def(6, 2) = -10; % state 6, right leg back/forward will be penalized
reward_def(6, 4) = -10; %   ...  , left leg    ...

reward_def(7, 2) = -10; % state 7, right leg back/forward will be penalized
reward_def(7, 4) = -10; %   ...  , left leg    ...

reward_def(8, 1) = -10; % state 8, right leg up/down will be penalized 
reward_def(8, 2) = 10;  % state 8, right leg back will be rewarded

reward_def(9, 1) = -10; % state 9, right leg up/down will be penalized
reward_def(9, 2) = -10; %   ...  ,  ...  back/forward  ...

reward_def(10, 2) = -10; % state 10, right leg back/forward will be penalized
reward_def(10, 4) = -10; %   ...  ,  left leg    ...

reward_def(11, 2) = -10; % state 11, right leg back/forward will be penalized
reward_def(11, 4) = -10; %   ...  ,  left leg    ...

reward_def(12, 1) = -10; % state 12, right leg up/down will be penalized 
reward_def(12, 2) = 10;  % state 12, right leg back will be rewarded

reward_def(13, 2) = -10; % state 13, right leg back/forward will be penalized
reward_def(13, 4) = -10; %   ...  ,  left leg    ...

reward_def(14, 3) = -10; % state 14, left leg up/down will be penalized
reward_def(14, 4) = 10;  % state 14, left leg back will be rewarded

reward_def(15, 3) = -10; % state 15, left leg up/down will be penalized
reward_def(15, 4) = 10;  % state 15, left leg back will be rewarded

reward_def(16, 2) = -10; % state 16, right leg back/forward will be penalized
reward_def(16, 4) = -10; %   ...  ,  left leg    ...

disp('The reward function is as follow:');
reward_def
disp('----------------------------------');

%% Task 2: Apply policy iteration
% sigma:state transition matrix
sigma = [ 2  4  5 13;
          1  3  6 14;
          4  2  7 15;
          3  1  8 16;
          6  8  1  9;
          5  7  2 10;
          8  6  3 11;
          7  5  4 12;
         10 12 13  5;
          9 11 14  6;
         12 10 15  7;
         11  9 16  8;
         14 16  9  1;
         13 15 10  2;
         16 14 11  3;
         15 13 12  4 ];

% Initialize
policy = ceil(rand(16, 1) * 4);
gamma = 0.7;    % discount factor
converge = 0;   % when not converge =1 
Initial_policy = policy;
iteration = 0;
  
while ~converge
    reward = zeros(1, 16);
    V_pi = sym('V_pi', [16 1]);
    A = zeros(16, 16);
    V = zeros(1, 16);   % V:value function
    for s = 1:16        % s:states, total 16 states
        reward(s) = reward_def(s, policy(s));
        A(s, sigma(s, policy(s))) = 1;
    end
    lin_eq = reward' + gamma * A * V_pi == V_pi;  % set linear equation
    V_pi_sol = vpasolve(lin_eq, V_pi);   % solve linear equation
    
    % Get numerical solution of vector V_pi from V_pi_sol, set V equal to it.
    V(1) = getfield(V_pi_sol,'V_pi1');
    V(2) = getfield(V_pi_sol,'V_pi2');
    V(3) = getfield(V_pi_sol,'V_pi3');
    V(4) = getfield(V_pi_sol,'V_pi4');
    V(5) = getfield(V_pi_sol,'V_pi5');
    V(6) = getfield(V_pi_sol,'V_pi6');
    V(7) = getfield(V_pi_sol,'V_pi7');
    V(8) = getfield(V_pi_sol,'V_pi8');
    V(9) = getfield(V_pi_sol,'V_pi9');
    V(10) = getfield(V_pi_sol,'V_pi10');
    V(11) = getfield(V_pi_sol,'V_pi11');
    V(12) = getfield(V_pi_sol,'V_pi12');
    V(13) = getfield(V_pi_sol,'V_pi13');
    V(14) = getfield(V_pi_sol,'V_pi14');
    V(15) = getfield(V_pi_sol,'V_pi15');
    V(16) = getfield(V_pi_sol,'V_pi16');
    
    % Policy iteration 
    for s = 1:16
        best_action = 0;
        max_reward = -inf;
        for a = 1:4
            next_state = sigma(s, a);
            current_reward = reward_def(s, a) + gamma * V(next_state);
            if current_reward > max_reward
                max_reward = current_reward;
                best_action = a;
            end
        end
        policy(s) = best_action;
    end
    
    % Check if converged or not
    if Initial_policy == policy
        converge = 1;
    else
        Initial_policy = policy;
    end
    iteration = iteration + 1;
end

disp('The number of total iteration is as follow:');
iteration

% Generate state sequence of walkshow
states = zeros(1, 16);
states(1) = Initial_state;   % give input state
for t = 1:(16-1)
    states(t+1) = sigma(states(t), policy(states(t)));
end

%displays a graphical ¡°cartoon¡± of the walking robot
walkshow(states)
end
