function par = Exercise1(k)

load('Data.mat');
%Input = (v;w) 2xn
%Output = (x;y;teta) 3xn, corrupted by zero mean gaussian noise

% Init Variables
k = 5;
% k = 2;
n = size(Input,2);
pMax = 6;
pOpt = [0,0];
minPosErr=inf;
minOrErr=inf;

for p=1:pMax
    % calc new Sum of Errors for each p
    sumPosErr = 0;
    sumOrErr = 0;  
    
    for K=1:k
        
        % Generate indices for k-fold validation
        % in each run 4 samples for training, 1 for testing
        % TeSI = Test Sample Indizes
        % TrSI = Training Sample Indizes
        TeSI = 1+(K-1)*round(n/k):K*round(n/k);
        TrSI = 1:n;   %or:  union(1:(K-1)*round(n/k),1+K*round(n/k):n);
        TrSI(TeSI) = [];
        
        % read train and test data
        vTrain=Input(1,TrSI)';
        wTrain=Input(2,TrSI)';
        
        vTest=Input(1,TeSI)';
        wTest=Input(2,TeSI)';
        
        xTrain=Output(1,TrSI)';
        yTrain=Output(2,TrSI)';
        tetaTrain=Output(3,TrSI)';
        
        xTest=Output(1,TeSI)';
        yTest=Output(2,TeSI)';
        tetaTest=Output(3,TeSI)';
        
        % linear Regression in Matrix notation
        X1Train=createX(vTrain,wTrain,p);
        X2Train=X1Train;
        X3Train=X1Train;
        
        X1Test=createX(vTest,wTest,p);
        X2Test=X1Test;
        X3Test=X1Test;
        
        % w =(X'X)^-1 X'y  | here "w" is "a" as in Ex description
        a1=(X1Train'*X1Train)\(X1Train'*xTrain);
        a2=(X2Train'*X2Train)\(X2Train'*yTrain);
        a3=(X3Train'*X3Train)\(X3Train'*tetaTrain);        
        
        % y = w*x | y = prediction
        xPred=X1Test*a1; 
        yPred=X2Test*a2; 
        tetaPred=X3Test*a3; 
       
        % Error calculation
        n_test = size(TeSI,2);
        posErr= sum(sqrt((xTest-xPred).^2+(yTest-yPred).^2))/n_test;
        orErr = sum(sqrt((tetaTest-tetaPred).^2))/n_test;
        sumPosErr=sumPosErr+posErr;
        sumOrErr=sumOrErr+orErr;
    end
    
    % chose p with smallest Error
    if(sumPosErr < minPosErr)
        minPosErr=sumPosErr;
        pOpt(1)=p;
    end
    if(sumOrErr < minOrErr)
        minOrErr=sumOrErr;
        pOpt(2)=p;
    end
end

 % use the entire dataset
 TrSI_en = 1:n;
 vTrain_en=Input(1,TrSI_en)';
 wTrain_en=Input(2,TrSI_en)';
 
 xTrain_en=Output(1,TrSI_en)';
 yTrain_en=Output(2,TrSI_en)';
 tetaTrain_en=Output(3,TrSI_en)';
 
 X1Train_optimal=createX(vTrain_en,wTrain_en,pOpt(1));
 X2Train_optimal=X1Train_optimal;
 X3Train_optimal=createX(vTrain_en,wTrain_en,pOpt(2));
 
 a1_optimal=(X1Train_optimal'*X1Train_optimal)\(X1Train_optimal'*xTrain_en);
 a2_optimal=(X2Train_optimal'*X2Train_optimal)\(X2Train_optimal'*yTrain_en);
 a3_optimal=(X3Train_optimal'*X3Train_optimal)\(X3Train_optimal'*tetaTrain_en);      

 par1=a1_optimal;
 par2=a2_optimal;
 par3=a3_optimal;
 
par={par1,par2,par3};
save('params','par');
%% simulate Robot
Simulate_robot(0, 0.05);
Simulate_robot(1, 0);
Simulate_robot(1, 0.05);
Simulate_robot(-1, -0.05);
%Simulate_robot(0.5,-0.03);
end
function X=createX(v,w,p)
n = size(v,1);
X = zeros(n, 1+p*3);
% X = [1,x1,x2,...,xm]
% first row only ones
X(:,1)=ones(n,1);
for i=1:p
    X(:,(i*3-1):(i*3+1)) = [(v).^i, (w).^i, ((v.*w)).^i]; % [v^p,w^p,(vw)^p,...repeat]
end
end