% Exercise-3 Zhen Zhou 03721400
% Task3 SimulateRobot function

function [ newstate, reward ] = SimulateRobot(state, action)

    %------------define reward-------------
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
    
    reward = reward_def(state, action);
    
    %--------define sigma---------
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
             15 13 12  4  ];
         
     newstate = sigma(state, action);
end

