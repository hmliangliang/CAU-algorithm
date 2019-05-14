function mse = MSE(traindata,traintarget,testdata,testtarget)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%此算法主要是计算均方误差
mode=NaiveBayes.fit(traindata,traintarget);
result=posterior(mode,testdata);
sum1=0;
for i=1:size(testdata,1)
    if testtarget(i,1)>size(result,2)
       result(i,testtarget(i,1))=0;
    end
    sum1=sum1+(1-result(i,testtarget(i,1)))^2;
end
mse=sum1/(size(testdata,1));
end

