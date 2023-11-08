
%Testing the trained Dkut Ai-content detection model
% By ERIC MULWA
close all
clear all
% Loading the trained model
load('Dkut_Ai_detection_model.mat', 'net');
% Preprocessing the test content in image form
testImage = imread('D:\Test dataset\6.PNG');
testImage = imresize(testImage, [224 224]);
% Classifying the test content in image form
prediction = classify(net, testImage);
% Displaying the predicted class
disp(['Predicted class: ' char(prediction)])
