clear
clc
close all

test = load('mnist_test.csv');
labels = test(:,1);
y = zeros(10,10000);
for i = 1:10000
    y(labels(i)+1,i) = 1;
end

images = test(:,2:785);
images = images/255;

images = images';

we34 = matfile('wfour.mat');
w34 = we34.w34;
we23 = matfile('wthree.mat');
w23 = we23.w23;
we12 = matfile('wtwo.mat');
w12 = we12.w12;
bi34 = matfile('bfour.mat');
b34 = bi34.b34;
bi23 = matfile('bthree.mat');
b23 = bi23.b23;
bi12 = matfile('btwo.mat');
b12 = bi12.b12;
%success = 0;
n = 10000;


a1 = images;
%z2 = w12*a1 + b12;
z2 = w12*a1 + b12;
%a2 = elu(z2);
a2 = elu_v2(z2);
z3 = w23*a2 + b23;
%a3 = elu(z3);
a3 = elu_v2(z3);
z4 = w34*a3 + b34;
%a4 = elu(z4); %Output vector
a4 = elu_v2(z4); %Output vector

% for i = 1:n
% out2 = elu(w2*images(:,i)+b2);
% out3 = elu(w3*out2+b3);
% out = elu(w4*out3+b4);
% big = 0;
% num = 0;
% for k = 1:10
%     if out(k) > big
%         num = k-1;
%         big = out(k);
%     end
% end
% 
% if labels(i) == num
%     success = success + 1;
% end
%     
% 
% end

tic

[M,I] = max(a4,[],1);
I_estimatedLabel = (I-1)';
success = sum(labels == I_estimatedLabel);

toc

fprintf('Accuracy: ');
fprintf('%f',success/n*100);
disp(' %');
fprintf("\nSuccess: %f",success);