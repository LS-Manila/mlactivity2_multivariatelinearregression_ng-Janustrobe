clear all; close all; clc

x = load('mlx.dat'); 
y = load('mly.dat');
figure, plot(x(:, 1), y,'o');
xlabel('batting average');
ylabel('years');
figure, plot(x(:, 2), y,'o');
xlabel('runs at bat');
ylabel('years');
m = length(y);

x = [ones(m, 1), x];

x_unscaled = x;

mu = mean(x);
sigma = std(x);
x(:,2) = (x(:,2) - mu(2))./ sigma(2);
x(:,3) = (x(:,3) - mu(3))./ sigma(3);

figure, plot(x(:, 2), y,'o');
xlabel('batting average');
ylabel('years');
figure, plot(x(:, 3), y,'o');
xlabel('runs');
ylabel('years');
figure; 
plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--', 'k--'};

alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.1, 1.3];
MAX_ITR = 50;
theta_grad_descent = zeros(size(x(1,:))); 

for i = 1:length(alpha)
    theta = zeros(size(x(1,:)))';
    J = zeros(MAX_ITR, 1);
    for num_iterations = 1:MAX_ITR
        J(num_iterations) = (0.5/m) .* (x * theta - y)' * (x * theta - y);
        
        grad = (1/m) .* x' * ((x * theta) - y);
        
        theta = theta - alpha(i) .* grad;
    end
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2)
    hold on
    
    if (alpha(i) == 1)
        theta_grad_descent = theta;
    end
end
legend('0.01','0.03','0.1', '0.3', '1', '1.1')
xlabel('Number of iterations')
ylabel('Cost J')

format long

theta_grad_descent

price_grad_desc = dot(theta_grad_descent, [1, (0.35 - mu(2))/sigma(2),...
                    (.198 - mu(3))/sigma(3)])


theta_normal = (x_unscaled' * x_unscaled)\x_unscaled' * y

price_normal = dot(theta_normal, [1, 0.35, 0.198])

