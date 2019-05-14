function mse = kMSE( data1,target1,k)
%UNTITLED Summary of this function goes here
%Detailed explanation goes here
%���㷨ִ�е��ǽ�����֤�Ĺ���
datanum=size(data1,1);%��¼���ݵ�������
%��ʼִ�н�������֤�ĳ���
leave=mod(datanum,k);
data=data1(1:(datanum-leave),:);
target=target1(1:(datanum-leave),:);
datanum=size(data1,1);%��¼���ݵ�������
foldnum=datanum/k;%��¼ÿһ�۵���������
error=[];
i=size(data,1);
for y=1:k
    testdata=data((i-y*foldnum+1):(i-y*foldnum+foldnum),:);
    testtarget=target((i-y*foldnum+1):(i-y*foldnum+foldnum),1);
    tempdata=data;%��ʱ��������
    temptarget=target;%��ʱ����Ŀ�����ǩ
    tempdata((i-y*foldnum+1):(i-y*foldnum+foldnum),:)=[];
    temptarget((i-y*foldnum+1):(i-y*foldnum+foldnum),:)=[];
    traindata=tempdata;
    traintarget=temptarget;
    %ѵ��Ŀ������������в���
    mode=NaiveBayes.fit(traindata,traintarget);
    result=posterior(mode,testdata);
    count=0;%��¼����
    [pro,dimension]=max(result,[],1);%pro��¼ÿһ�����ĸ��ʣ�dimension��¼�����ʶ�Ӧ�����ǩ
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

