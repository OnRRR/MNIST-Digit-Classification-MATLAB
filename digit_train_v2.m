clear
clc
close all

data = load('mnist_train.csv');

labels = data(:,1);
y = zeros(10,60000); %Correct outputs vector
for i = 1:60000
    y(labels(i)+1,i) = 1;
end

images = data(:,2:785);
images = images/255;

images = images'; %Input vectors

hn1 = 80; %Number of neurons in the first hidden layer
hn2 = 60; %Number of neurons in the second hidden layer

%Initializing weights and biases
w12 = randn(hn1,784)*sqrt(2/784);
w23 = randn(hn2,hn1)*sqrt(2/hn1);
w34 = randn(10,hn2)*sqrt(2/hn2);
b12 = randn(hn1,1);
b23 = randn(hn2,1);
b34 = randn(10,1);

%learning rate
eta = 0.0058;

epochs = 50;

m = 10; %Minibatch size

for k = 1:epochs %Outer epoch loop
    
    batches = 1;
    
    tic
    
    for j = 1:60000/m
        
        %errortot4 = zeros(10,1);
        %errortot3 = zeros(hn2,1);
        %errortot2 = zeros(hn1,1);
        %         
        %grad4 = zeros(size(w34));
        %grad3 = zeros(size(w23));
        %grad2 = zeros(size(w12));
        %         
        %for i = batches:batches+m-1 %Loop over each minibatch
        
        %Feed forward
        %a1 = images(:,i);
        a1 = images(:,batches:batches+m-1);
        z2 = w12*a1 + b12;
        %a2 = elu(z2);
        a2 = elu_v2(z2);
        z3 = w23*a2 + b23;
        %a3 = elu(z3);
        a3 = elu_v2(z3);
        z4 = w34*a3 + b34;
        %a4 = elu(z4); %Output vector
        a4 = elu_v2(z4); %Output vector
        
        %backpropagation
        %error4 = (a4-y(:,i)).*elup(z4);
        error4 = ((a4-y(:,batches:batches+m-1))/m).*elup_v2(z4);
        %error3 = (w34'*error4).*elup(z3);
        error3 = (w34'*error4).*elup_v2(z3);
        %error2 = (w23'*error3).*elup(z2);
        error2 = (w23'*error3).*elup_v2(z2);
        
        %errortot4 = errortot4 + error4;
        %errortot3 = errortot3 + error3;
        %errortot2 = errortot2 + error2;
        %grad4 = grad4 + error4*a3';
        %grad3 = grad3 + error3*a2';
        %grad2 = grad2 + error2*a1';
        
        grad4 = error4*a3';
        grad3 = error3*a2';
        grad2 = error2*a1';
        %
        %end
        %
        %Gradient descent
        %w34 = w34 - eta/m*grad4;
        %w23 = w23 - eta/m*grad3;
        %w12 = w12 - eta/m*grad2;
        w34 = w34 - eta*grad4;
        w23 = w23 - eta*grad3;
        w12 = w12 - eta*grad2;
        %b34 = b34 - eta/m*errortot4;
        %b23 = b23 - eta/m*errortot3;
        %b12 = b12 - eta/m*errortot2;
        b34 = b34 - eta*sum(error4,2);
        b23 = b23 - eta*sum(error3,2);
        b12 = b12 - eta*sum(error2,2);
        
        batches = batches + m;
        
    end
    fprintf('Epochs:');
    disp(k) %Track number of epochs
    [images,y] = shuffle(images,y); %Shuffles order of the images for next epoch
    
    toc
    
end

disp('Training done!')
%Saves the parameters
save('wfour.mat','w34');
save('wthree.mat','w23');
save('wtwo.mat','w12');
save('bfour.mat','b34');
save('bthree.mat','b23');
save('btwo.mat','b12');