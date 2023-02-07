% Exercise-3 Zhen Zhou 03721400
% Task3 Apply Q-learning

function [ ] = WalkQLearning(state)
%% Initialize parameters
Q = zeros(16, 4);
actions = 1:1:4;
s = 12;             % arbitrary initial state
I = 3000;           % maximal learning iterations
alpha = 0.4;        % learning rate
epsilon = 0.25;     % probability of exploring randomly
gamma = 0.7;        % discount factor
iteration = 0;

%% Q-Learning process
while iteration < I
    [~, best_action] = max(Q(s, :));
    % Make a random decision based on epsilon. 
    random_seed = rand;
    random_decision = sum(random_seed >= cumsum([0, 1-epsilon, epsilon]));
    if random_decision == 1
        a = best_action;
    else
        a = datasample(actions, 1);
    end
    
    % find next state and reward 
   [ next_state, r ] = SimulateRobot(s, a);
 
    % Update Q
    max_a_prime = max(Q(next_state, :));
    Q(s, a) = Q(s, a) + alpha * (r + gamma * max_a_prime - Q(s, a));
    s = next_state;
    iteration = iteration + 1;
end

disp('The number of total iteration is as follow:');
iteration

% Generate state sequence of walkshow
states = zeros(1, 16);
states(1) = state;
for t = 1:(16-1)
    [~, best_action] = max(Q(states(t), :));
    [states(t+1), ~] = SimulateRobot(states(t), best_action);
end

%displays a graphical ¡°cartoon¡± of the walking robot
walkshow(states)

end