function Exercise3_kmeans( Data1, Data2, Data3, init1, init2, init3, k)
%% K-Means Algo[labels_i] = k-means(Data1,init1,k);
[labels1] = k_means(Data1,init1,k);
[labels2] = k_means(Data2,init2,k);
[labels3] = k_means(Data3,init3,k);

%% Plot clusters produced by K-means
plot_clusters(Data1,labels1,'k-means_l')
plot_clusters(Data2,labels2,'k-means_o')
plot_clusters(Data3,labels3,'k-means_x')
end

function [labels] = k_means(Data,Init,k)
% init variables
Data = reshape(Data,[600 3]);
Init_cluster = Init;
N = size(Data,1);
J_old = inf;
decrement_threshold = 10e-6;
decrement = inf;
%steps=0;

while (decrement > decrement_threshold)
    % initialize min distances and class labels in each step
    dist_min = zeros(N,1);
    labels = zeros(N,1);

    % E-Step: find the closest class and label them
    for i=1:N
        %initialize distance
        dist = zeros(k,1);
        
        for j=1:k
            dist(j) = euk_dist(Data(i,:),Init_cluster(j,:));
        end

        [dist_min(i), labels(i)] = min(dist);
    end
    
    % M-step: update new mean of each cluster
    for j=1:k
        Init_cluster(j,:) = mean(Data(labels==j,:));
    end
 
    % Calculate the total distortion
    J = sum(dist_min);
    decrement = J_old - J;
    J_old = J;
    %steps = steps +1;
    %plot_clusters(Data,cluster,'test_k_means')
end
end