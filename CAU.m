%XU S L， WANG J H． Classification Algorithm Combined with Unsupervised Learning for Data Stream． Pattern Recognition and Artificial Intelligence， 2016， 29(7): 665 － 672．

data=waveform;%data set is waveform
theta=0.2;%判断是否发生概念漂移的标准数
k=5;%子分类器数目的上限
winsize=200;%滑动窗口的大小
e=0.0000001;%e为很小的常数，防止在计算权值的过程中除数为0
col=size(data,2);%数据的维数
ensemble=struct('traindata',[],'traintarget',[],'weight',[]);%集成式分类器结构
acc=[];%保存每次测试结果的准确率
cc=0;%记录测试的次数
tic;

for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %获取训练集和测试集
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));
        traintarget=data((i-2*winsize+1):(i-winsize),col);
        testdata=data((i-winsize+1):i,1:(col-1));
        testtarget=data((i-winsize+1):i,col);
        klabel=length(unique(traintarget(:,1)));%klabel代表训练集中不同类标签的数目
        num=max(unique(traintarget(:,1)));%类标签的最大值
        mser=(1/klabel)*((1-(1/klabel))^2);%计算随机分类的均方误差
        %训练一个新分类器
        temp=struct('traindata',[],'traintarget',[],'weight',[]);
        temp.traindata=traindata;
        temp.traintarget=traintarget;
        if (size(ensemble,2)==1)&&(isempty(ensemble(1).traindata)==1)%分类器中无子分类器
           %先训练后一个子分类器
           ensemble(1).traindata=[ensemble(1).traindata;traindata];
           ensemble(1).traintarget=[ensemble(1).traintarget;traintarget];
           ensemble(1).weight=1/(mser+e);
        else
            result=[];%记录每个分类器对数据块的分类结果,行代表事例，列代表分类器
            weight=zeros(size(traindata,1),num);
            for j=1:size(ensemble,2)
                model=ClassificationTree.fit(ensemble(j).traindata,ensemble(j).traintarget);
                res=predict(model,traindata);
                result=[result,res];
            end
            %统计投票结果
            for ri=1:size(result,1)
                for rj=1:size(result,2)
                    if result(ri,rj)<=num
                       weight(ri,result(ri,rj))=weight(ri,result(ri,rj))+ensemble(rj).weight;
                    end
                end
            end
            [c1,label]=max(weight,[],2);%c为投票的累计权重值，label为最终的投票结果
            count=0;
            for la=1:size(label,1)
                if label(la,1)==traintarget(la,1)
                    count=count+1;
                end
            end
            right1=count/(size(label,1));%right1为集成式分类器对数据集的分类结果
            %根据子分类器对当前数据集的分类结果更新分类器的权值
            for ei=1:size(ensemble,2)
                cmodel=NaiveBayes.fit(ensemble(ei).traindata,ensemble(ei).traintarget);
                pos=posterior(cmodel,traindata);
                sumi=0;
                QA=unique(traintarget(:,1));
                for si=1:size(traindata,1)
                    sp=find(QA==traintarget(si,1));
                    if sp>size(pos,2)
                        pos(si,sp)=0;
                    end
                    sumi=sumi+(1-pos(si,sp))^2;
                end
                mse=sumi/(size(traindata,1));
                ensemble(ei).weight=1/(mser+mse+e);
                %用当前训练集去训练每一个分类器
                ensemble(ei).traindata=[ensemble(ei).traindata;traindata];
                ensemble(ei).traintarget=[ensemble(ei).traintarget;traintarget];
            end
            
            
            %进行属性约简，聚类
            %data1防止traindata遭到破坏，因为后面发生概念漂移时，需要利用traindata训练新的分类器
            data1=zscore(traindata);
            data0=Reduction(data1,traintarget);
            canre=size(traindata,2)-size(data0,2);
            disp(['第',num2str(cc+1),'次数据集能够约简掉的维数为：',num2str(canre)]);
            %tempdata保存各个类别的数据
            tempdata=struct('data',[]);
            tempdata(num).data=[];
            for ts=1:size(data0,1)
                tempdata(traintarget(ts,1)).data=[tempdata(traintarget(ts,1)).data;data0(ts,:)];
            end
            center=[];%保存初始中心点
            %消除tempdata中的空项
            hi=1;
            while hi<=size(tempdata,2)
                if isempty(tempdata(hi).data)==1
                    tempdata(hi)=[];
                else
                    hi=hi+1;
                end
            end            
            %计算出初始中心点
            for ti=1:size(tempdata,2)
                gdata=mean(tempdata(ti).data);%gdata保存每个类标数据点的平均值
                center=[center;gdata];     
            end
            %利用k-means算法进行聚类
            kresult=kmeans(data0,klabel,'emptyaction','drop','start',center);
            kweight=zeros(klabel,num);
            for qi=1:size(kresult,1)
                kweight(kresult(qi,1),traintarget(qi,1))=kweight(kresult(qi,1),traintarget(qi,1))+1;
            end
            
            [c1,df]=max(kweight,[],2);%建立起聚类的类标签和真实类标签之间的关系,df保存聚类类标签和真实类标签对应的关系
            %将据类类标签替换成真实类标签
            for c=1:size(df,1)
                kresult(c,1)=df(kresult(c,1),1);
            end
            %计算聚类结果的准确率
            kcount=0;
            for n=1:size(kresult,1)
                if kresult(n,1)==traintarget(n,1)
                    kcount=kcount+1;
                end
            end
            right2=kcount/(size(kresult,1));
            
            %判断是否发生概念漂移
            if abs(right1-right2)>theta %发生概念漂移
                disp('发生概念漂移');
               [v,u]=min([ensemble.weight]);
               ensemble(u)=[];%删除权值最小的分类器
               temp.weight=1/(mse+e);
               ensemble=[ensemble,temp];
            else %未发生概念漂移
                if size(ensemble,2)<k %分类器系统未满
                    temp.weight=1/(mse+e);
                    ensemble=[ensemble,temp];
                end
            end           
        end
        %进行测试
        cc=cc+1;
        ccount=0;
        cresult=[];
        for ci=1:size(ensemble,2)
           cmodel=ClassificationTree.fit(ensemble(ci).traindata,ensemble(ci).traintarget);
           cres=predict(cmodel,testdata);
           cresult=[cresult,cres];
        end
        cweight=zeros(size(cresult,1),num);%记录投票权值
        for cx=1:size(cresult,1)
            for cy=1:size(cresult,2)
                if cresult(cx,cy)<=num
                  cweight(cx,cresult(cx,cy))=cweight(cx,cresult(cx,cy))+ensemble(cy).weight;
                end
            end
        end
        %获得最终投票值
        [c3,clabel]=max(cweight,[],2);
        %统计正确率
        for y=1:size(clabel,1)
            if clabel(y,1)==testtarget(y,1)
                ccount=ccount+1;
            end
        end
        cr=ccount/(size(clabel,1));%计算当前测试正确率
        disp(['第',num2str(cc),'次测试结果的准确率为：',num2str(cr)]);
        acc=[acc,cr];
    end
end
lastacc=mean(acc);
disp(['CAU算法在当前数据集上测试结果的平均准确率为：',num2str(lastacc)]);
toc;

