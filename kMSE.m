function mse = kMSE( data1,target1,k)
%UNTITLED Summary of this function goes here
%Detailed explanation goes here
%此算法执行的是交叉验证的过程
datanum=size(data1,1);%记录数据的总数量
%开始执行交叉已验证的程序
leave=mod(datanum,k);
data=data1(1:(datanum-leave),:);
target=target1(1:(datanum-leave),:);
datanum=size(data1,1);%记录数据的总数量
foldnum=datanum/k;%记录每一折的数据数量
error=[];
i=size(data,1);
for y=1:k
    testdata=data((i-y*foldnum+1):(i-y*foldnum+foldnum),:);
    testtarget=target((i-y*foldnum+1):(i-y*foldnum+foldnum),1);
    tempdata=data;%临时保存数据
    temptarget=target;%临时保存目标类标签
    tempdata((i-y*foldnum+1):(i-y*foldnum+foldnum),:)=[];
    temptarget((i-y*foldnum+1):(i-y*foldnum+foldnum),:)=[];
    traindata=tempdata;
    traintarget=temptarget;
    %训练目标分类器，进行测试
    mode=NaiveBayes.fit(traindata,traintarget);
    result=posterior(mode,testdata);
    count=0;%记录错误
    [pro,dimension]=max(result,[],1);%pro记录每一列最大的概率，dimension记录最大概率对应的类标签
    MSE=0;
    for j=1:size(pro,1)
        if testtarget(j,1)>size(pro,2)
            pro(j,testtarget(j,1))=0;
        end
        MSE=MSE+((1-pro(j,testtarget(j,1)))^2);
    end
    MSE=MSE/foldnum;
    error=[error,MSE];
end
mse=sum(error)/k;
end

