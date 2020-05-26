%Convolutional Neural Networks (CNN)
clear all; close all; clc;
%preprocessing the training data
[trainData,trainLabels,valData,valLabels,testData,testLabels]=extractCifar10('C:\Users\R\Desktop\Winter 2020\ECE 172A\HW4\data\Problem_3');
%defining ConvNet architecture, instantiating a Layers object
imageSize=[32 32 3]; %32x32x3 color image
layers=[
    imageInputLayer(imageSize) %32x32x3
    
    convolution2dLayer(3,16) %16 3x3 filters
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2) %downsample by 2
    
    convolution2dLayer(3,32) %32 3x3 filters
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2) %downsample by 2
    
    convolution2dLayer(3,16) %16 3x3 filters
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(10) %10 units
    softmaxLayer
    classificationLayer
];
%defining parameters for stochastic gradient descent
options=trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',128, ...
    'ValidationData',{valData,valLabels}, ...
    'ValidationFrequency',352, ... %45000/1128=351.56=352
    'Plots','training-progress');
%% train the model
net = trainNetwork(trainData,trainLabels,layers,options);
%% image classification results
%comparing classification result with solution
    trainFeatures = activations(net,trainData,14); %softmax layer=14
    %trainLabels classified from [0,9], so need to map to [1,10]
    cmp=zeros(1,size(trainLabels,1));
    prediction=zeros(1,size(trainLabels,1));
    for i=1:1:size(trainLabels,1)
        [m,idx]=max(trainFeatures(i,:));
        prediction(i)=idx;
        cmp(i)=((uint8(trainLabels(i,1)))==idx); %uint8() automatically increments trainLabels by 1
    end
    figure; stem(cmp(1:20)); ylim([0,2]); title('Binary Classification Result Comparision with Solution');
%correct classification
    %three different images with bar plot of softmax layer output
    idx=[2 10 14];
    for i=1:1:size(idx,2)
        figure; imagesc(trainData(:,:,:,idx(i))); title(sprintf('Image %d',idx(i)));
        figure; bar(trainFeatures(idx(i),:)); title(sprintf('Softmax %d',idx(i)));
    end
%incorrect classification
    idx=[5 8 11];
    for i=1:1:size(idx,2)
        figure; imagesc(trainData(:,:,:,idx(i))); title(sprintf('Image %d',idx(i)));
        figure; bar(trainFeatures(idx(i),:)); title(sprintf('Softmax %d',idx(i)));
    end
%confusion matrix (10x10) for complete test set (10 classes)
    %trainFeatures = activations(net,trainData,2);
    %{
    svm = fitcecoc(trainFeatures,trainLabels);
    testFeatures = activations(net,testData,14);
    testPredictions = predict(svm,testFeatures);
    plotconfusion(testLabels,testPredictions);
    %}
    %can just use confusionmat() and imshow()to plot
    con_mat=confusionmat(double(trainLabels),prediction');
    figure; imagesc(con_mat); colorbar; title('Confusion Matrix');
    %figure; confusionchart(con_mat); %undefined function 'confusionchart'
%% image pre-processing
%original image
    im=imread('CNN_test.bmp');
    [YPred, scores]=classify(net, im);
    double(YPred)
    max(scores)
    figure; bar(scores);
%salt and pepper noise
    sp=imnoise(im,'salt & pepper');
    figure; imagesc(sp); title('Image with Salt & Pepper Noise');
    [YPred, scores]=classify(net, sp);
    double(YPred)
    max(scores)
    figure; bar(scores);
%gaussian noise
    gn=imnoise(im,'gaussian');
    figure; imagesc(gn); title('Image with Gaussian Noise');
    [YPred, scores]=classify(net, gn);
    double(YPred)
    max(scores)
    figure; bar(scores);
%gaussian smoothing
    gs=imgaussfilt(im,2);
    figure; imagesc(gs); title('Image with Gaussian Smoothing, \sigma = 2');
    [YPred, scores]=classify(net, gs);
    double(YPred)
    max(scores)
    figure; bar(scores);
%sharpen image
    si=imsharpen(im);
    figure; imagesc(si); title('Image with Sharpened Image');
    [YPred, scores]=classify(net, si);
    double(YPred)
    max(scores)
    figure; bar(scores);
%sharpen image by 3
    si=imsharpen(im,'Amount', 3);
    figure; imagesc(si); title('Image with Sharpened Image');
    [YPred, scores]=classify(net, si);
    double(YPred)
    max(scores)
    figure; bar(scores);