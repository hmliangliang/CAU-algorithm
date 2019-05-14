%XU S L�� WANG J H�� Classification Algorithm Combined with Unsupervised Learning for Data Stream�� Pattern Recognition and Artificial Intelligence�� 2016�� 29(7): 665 �� 672��

data=waveform;%data set is waveform
theta=0.2;%�ж��Ƿ�������Ư�Ƶı�׼��
k=5;%�ӷ�������Ŀ������
winsize=200;%�������ڵĴ�С
e=0.0000001;%eΪ��С�ĳ�������ֹ�ڼ���Ȩֵ�Ĺ����г���Ϊ0
col=size(data,2);%���ݵ�ά��
ensemble=struct('traindata',[],'traintarget',[],'weight',[]);%����ʽ�������ṹ
acc=[];%����ÿ�β��Խ����׼ȷ��
cc=0;%��¼���ԵĴ���
tic;

for i=1:size(data,1)
    if mod(i,2*winsize)==0
        %��ȡѵ�����Ͳ��Լ�
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));
        traintarget=data((i-2*winsize+1):(i-winsize),col);
        testdata=data((i-winsize+1):i,1:(col-1));
        testtarget=data((i-winsize+1):i,col);
        klabel=length(unique(traintarget(:,1)));%klabel����ѵ�����в�ͬ���ǩ����Ŀ
        num=max(unique(traintarget(:,1)));%���ǩ�����ֵ
        mser=(1/klabel)*((1-(1/klabel))^2);%�����������ľ������
        %ѵ��һ���·�����
        temp=struct('traindata',[],'traintarget',[],'weight',[]);
        temp.traindata=traindata;
        temp.traintarget=traintarget;
        if (size(ensemble,2)==1)&&(isempty(ensemble(1).traindata)==1)%�����������ӷ�����
           %��ѵ����һ���ӷ�����
           ensemble(1).traindata=[ensemble(1).traindata;traindata];
           ensemble(1).traintarget=[ensemble(1).traintarget;traintarget];
           ensemble(1).weight=1/(mser+e);
        else
            result=[];%��¼ÿ�������������ݿ�ķ�����,�д����������д��������
            weight=zeros(size(traindata,1),num);
            for j=1:size(ensemble,2)
                model=ClassificationTree.fit(ensemble(j).traindata,ensemble(j).traintarget);
                res=predict(model,traindata);
                result=[result,res];
            end
            %ͳ��ͶƱ���
            for ri=1:size(result,1)
                for rj=1:size(result,2)
                    if result(ri,rj)<=num
                       weight(ri,result(ri,rj))=weight(ri,result(ri,rj))+ensemble(rj).weight;
                    end
                end
            end
            [c1,label]=max(weight,[],2);%cΪͶƱ���ۼ�Ȩ��ֵ��labelΪ���յ�ͶƱ���
            count=0;
            for la=1:size(label,1)
                if label(la,1)==traintarget(la,1)
                    count=count+1;
                end
            end
            right1=count/(size(label,1));%right1Ϊ����ʽ�����������ݼ��ķ�����
            %�����ӷ������Ե�ǰ���ݼ��ķ��������·�������Ȩֵ
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
                %�õ�ǰѵ����ȥѵ��ÿһ��������
                ensemble(ei).traindata=[ensemble(ei).traindata;traindata];
                ensemble(ei).traintarget=[ensemble(ei).traintarget;traintarget];
            end
            
            
            %��������Լ�򣬾���
            %data1��ֹtraindata�⵽�ƻ�����Ϊ���淢������Ư��ʱ����Ҫ����traindataѵ���µķ�����
            data1=zscore(traindata);
            data0=Reduction(data1,traintarget);
            canre=size(traindata,2)-size(data0,2);
            disp(['��',num2str(cc+1),'�����ݼ��ܹ�Լ�����ά��Ϊ��',num2str(canre)]);
            %tempdata���������������
            tempdata=struct('data',[]);
            tempdata(num).data=[];
            for ts=1:size(data0,1)
                tempdata(traintarget(ts,1)).data=[tempdata(traintarget(ts,1)).data;data0(ts,:)];
            end
            center=[];%�����ʼ���ĵ�
            %����tempdata�еĿ���
            hi=1;
            while hi<=size(tempdata,2)
                if isempty(tempdata(hi).data)==1
                    tempdata(hi)=[];
                else
                    hi=hi+1;
                end
            end            
            %�������ʼ���ĵ�
            for ti=1:size(tempdata,2)
                gdata=mean(tempdata(ti).data);%gdata����ÿ��������ݵ��ƽ��ֵ
                center=[center;gdata];     
            end
            %����k-means�㷨���о���
            kresult=kmeans(data0,klabel,'emptyaction','drop','start',center);
            kweight=zeros(klabel,num);
            for qi=1:size(kresult,1)
                kweight(kresult(qi,1),traintarget(qi,1))=kweight(kresult(qi,1),traintarget(qi,1))+1;
            end
            
            [c1,df]=max(kweight,[],2);%�������������ǩ����ʵ���ǩ֮��Ĺ�ϵ,df����������ǩ����ʵ���ǩ��Ӧ�Ĺ�ϵ
            %���������ǩ�滻����ʵ���ǩ
            for c=1:size(df,1)
                kresult(c,1)=df(kresult(c,1),1);
            end
            %�����������׼ȷ��
            kcount=0;
            for n=1:size(kresult,1)
                if kresult(n,1)==traintarget(n,1)
                    kcount=kcount+1;
                end
            end
            right2=kcount/(size(kresult,1));
            
            %�ж��Ƿ�������Ư��
            if abs(right1-right2)>theta %��������Ư��
                disp('��������Ư��');
               [v,u]=min([ensemble.weight]);
               ensemble(u)=[];%ɾ��Ȩֵ��С�ķ�����
               temp.weight=1/(mse+e);
               ensemble=[ensemble,temp];
            else %δ��������Ư��
                if size(ensemble,2)<k %������ϵͳδ��
                    temp.weight=1/(mse+e);
                    ensemble=[ensemble,temp];
                end
            end           
        end
        %���в���
        cc=cc+1;
        ccount=0;
        cresult=[];
        for ci=1:size(ensemble,2)
           cmodel=ClassificationTree.fit(ensemble(ci).traindata,ensemble(ci).traintarget);
           cres=predict(cmodel,testdata);
           cresult=[cresult,cres];
        end
        cweight=zeros(size(cresult,1),num);%��¼ͶƱȨֵ
        for cx=1:size(cresult,1)
            for cy=1:size(cresult,2)
                if cresult(cx,cy)<=num
                  cweight(cx,cresult(cx,cy))=cweight(cx,cresult(cx,cy))+ensemble(cy).weight;
                end
            end
        end
        %�������ͶƱֵ
        [c3,clabel]=max(cweight,[],2);
        %ͳ����ȷ��
        for y=1:size(clabel,1)
            if clabel(y,1)==testtarget(y,1)
                ccount=ccount+1;
            end
        end
        cr=ccount/(size(clabel,1));%���㵱ǰ������ȷ��
        disp(['��',num2str(cc),'�β��Խ����׼ȷ��Ϊ��',num2str(cr)]);
        acc=[acc,cr];
    end
end
lastacc=mean(acc);
disp(['CAU�㷨�ڵ�ǰ���ݼ��ϲ��Խ����ƽ��׼ȷ��Ϊ��',num2str(lastacc)]);
toc;

