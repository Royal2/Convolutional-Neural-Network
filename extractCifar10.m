%{
This function extracts the CIFAR-10 dataset into a useful form for training your ConvNets
 
Inputs:
path_to_files: Path to the folder where you have extracted the CIFAR-10 tar.gz file 

Outputs:
trainData: 4-D array of size (32x32x3x<number of training samples>) containing images from the training data  
trainLabels: 1-D array of size (<number of training samples>) containing the corresponding class labels
valData: 4-D array of size (32x32x3x<number of validation samples>) containing images from the validation data
valLabels: 1-D array of size (<number of validation samples>) containing the corresponding class labels
testData: 4-D array of size (32x32x3x<number of test samples>) containing images from the test data
testLabels: 1-D array of size (<number of test samples>) containing the corresponding class labels

%}
function [trainData,trainLabels,valData,valLabels,testData,testLabels] = extractCifar10(path_to_files)
trainData = [];
testData = [];
trainLabels = [];
testLabels = [];

for k = 1:5
    filename = strcat(path_to_files,'/data_batch_',mat2str(k));
    load(filename);
    data = reshape(data,[size(data,1),32,32,3]);
    data = permute(data,[3,2,4,1]);
    trainData = cat(4,trainData,data);
    trainLabels = [trainLabels;labels];
end

load(strcat(path_to_files,'/test_batch'));
data = reshape(data,[size(data,1),32,32,3]);
data = permute(data,[3,2,4,1]);
testData = cat(4,testData,data);
testLabels = categorical(labels);

rndInd = randperm(size(trainData,4));
trainData = trainData(:,:,:,rndInd);
trainLabels = trainLabels(rndInd);
valData = trainData(:,:,:,45001:end);
valLabels = categorical(trainLabels(45001:end));
trainData = trainData(:,:,:,1:45000);
trainLabels = categorical(trainLabels(1:45000));

end