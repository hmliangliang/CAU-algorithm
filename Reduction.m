function data1 = Reduction( data,target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%���㷨��Ҫ�ǽ�������Լ��
klabel=length(unique(target(:,1)));%���ǩ��
ii=1;%ii�ĳ�ʼֵ���������Ե��к�
while ii<=size(data,2)
     %�������ĵ�   
     num=max(unique(target(:,1)));
     point=struct('data',[]);%���ĵ�
     point(num).data=[];
     point(num).target=[];
     for ni=1:size(data,1)
         point(target(ni,1)).data=[point(target(ni,1)).data;data(ni,:)];
     end
     %����point�ṹ���еĿ�Ԫ��
     i=1;
     while i<=size(point,2)
           if isempty(point(i).data)==1 
              point(i)=[];
           else
              i=i+1;
           end
     end
     center=[];%�������ĵ�
     for si=1:size(point,2)%��ƽ�����ĵ�
         temp=mean(point(si).data);
         center=[center;temp];%������ĵ�
     end
     kresult=kmeans(data,klabel,'emptyaction','drop','start',center);%����k-means�㷨���о���
     
     %ȷ����������Ӧ����ʵ����
     kweight=zeros(klabel,num);
     for j=1:size(kresult,1)
         kweight(kresult(j,1),target(j,1))=kweight(kresult(j,1),target(j,1))+1;
     end
     [c,d]=max(kweight,[],2);%c����ÿ�е����ֵ��d����ÿ�ж�Ӧ�����ǩ
     %�Ѿ�������ǩ��дΪʵ�ʶ�Ӧ�����ǩ
     for it=1:size(kresult,1)
         kresult(it,1)=d(kresult(it,1),1);
     end
     %����������ȷ��
     kcount=0;
     for tt=1:size(kresult,1)
         if kresult(tt,1)==target(tt,1)
             kcount=kcount+1;
         end
     end
     right0=kcount/(size(kresult,1));
     
     %ɾ����iiά���ԣ��൱��Լ��
     center(:,ii)=[];%%ɾ����iiά�������Ժ������
     datai=data;%��ֹԭ�������⵽�ƻ�
     datai(:,ii)=[];%ɾ����iiά�������Ժ������
     kresult2=kmeans(datai,klabel,'emptyaction','drop','start',center);
     %ȷ����������Ӧ����ʵ���ǩ
     kweight2=zeros(klabel,num);
     for ai=1:size(kweight2,1)
         kweight2(kresult2(ai,1),target(ai,1))=kweight2(kresult2(ai,1),target(ai,1))+1;
     end
     [c,d2]=max(kweight2,[],2);%���ö���ͶƱ�ķ�ʽ�����������ǩ��c�������ֵ��d2�������ֵ��Ӧ�����ǩ
     %�޸ľ�����Ϊ��Ӧ����ʵ���ǩ
     for yi=1:size(d2,1)
         kresult2(yi,1)=d2(kresult2(yi,1),1);
     end
     %ͳ����ȷ��
     kcount2=0;
     for yj=1:size(kresult2,1)
         if kresult2(yj,1)==target(yj,1)
             kcount2=kcount2+1;
         end
     end
     right2=kcount2/(size(target,1));%������ȷ��
     
     %�ж������Ƿ����Լ��     
     if right2>=right0 %˵����ǰ���Կ���Լ��
         data(:,ii)=[];
     else%˵����ǰ���Բ���Լ��
         ii=ii+1;
     end
     
end %endwhile  ii
data1=data;

end %end function

