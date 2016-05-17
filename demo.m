
%% load data: 200 data points(100 data points for each class), two pre-trained prototyes, a new data point 
close all; clear;
load ./demo_data.mat
%% Plot data and prototypes
figure;
plot(Data(1:100,1), Data(1:100,2), 'gs'); % class 1
hold on; 
plot(Data(101:200,1), Data(101:200,2), 'rx'); % class 2
% Prototypes
plot(Protos(1,1),Protos(1,2),'rs', 'MarkerSize',20,'MarkerEdgeColor','green','Linewidth',2);
plot(Protos(2,1),Protos(2,2),'rs', 'MarkerSize',20,'MarkerEdgeColor','red','Linewidth',2);
% new data point
plot(newData(:,1), newData(:,2), 'x', 'MarkerSize',15)

%% run cp
oRe = cp_with_protots(Data, Labels, newData, trueLabel, Protos, protoLabel)