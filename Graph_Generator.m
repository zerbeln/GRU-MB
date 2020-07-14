%Rover Domain Graphing

clear all; close all; clc

%% Test Parameters
stat_runs = 30;
generations = 1000;

%% Input from Text Files

four_deep_data = importdata('4-Deep/Output_Data/Training_Fitness.csv');
five_deep_data = importdata('5-Deep/Output_Data/Training_Fitness.csv');
six_deep_data = importdata('6-Deep/Output_Data/Training_Fitness.csv');
fifteen_deep_data = importdata('15-Deep/Output_Data/Training_Fitness.csv');
twenty_one_deep_data = importdata('21-Deep/Output_Data/Training_Fitness.csv');

%% Data Analysis

four_fitness = mean(four_deep_data.data, 1);
four_error = std(four_deep_data.data, 0, 1);

five_fitness = mean(five_deep_data.data, 1);
five_error = std(five_deep_data.data, 0, 1);

six_fitness = mean(six_deep_data.data, 1);
six_error = std(six_deep_data.data, 0, 1)/sqrt(stat_runs);

fifteen_fitness = mean(fifteen_deep_data.data, 1);

twenty_one_fitness = mean(twenty_one_deep_data.data, 1);

%% Graph Generator
color1 = [114, 147, 203]/255;
color2 = [132, 186, 91]/255;
color3 = [211, 94, 96]/255;
color4 = [128, 133, 133]/255;
color5 = [144, 103, 167]/255;
alpha = 0.3;

X = [1:generations];
x_axis = [X, fliplr(X)];
spacing = 20;

figure(1)
hold on
% 4-Deep
plot(X(1:spacing:end), four_fitness(1:spacing:end), '->', 'Color', color1, 'Linewidth', 1.5)

% 5-Deep
plot(X(1:spacing:end), five_fitness(1:spacing:end), '-^', 'Color', color2, 'Linewidth', 1.5)

% 6-Deep
plot(X(1:spacing:end), six_fitness(1:spacing:end), '-d', 'Color', color3, 'Linewidth', 1.5)

% 15-Deep
plot(X(1:spacing:end), fifteen_fitness(1:spacing:end), '-d', 'Color', color4, 'Linewidth', 1.5)

% 21-Deep
plot(X(1:spacing:end), twenty_one_fitness(1:spacing:end), '-d', 'Color', color5, 'Linewidth', 1.5)


% Graph Options
box on
legend('4 Deep', '5 Deep', '6 Deep', '15 Deep', '21 Deep', 'Orientation', 'horizontal')
%title('Clusters, Coupling 3')
xlabel('Generations')
ylabel('Correctly Sorted Sequences (%)')
