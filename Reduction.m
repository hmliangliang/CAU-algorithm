function data1 = Reduction( data,target )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%本算法主要是进行属性约简
klabel=length(unique(target(:,1)));%类标签数
ii=1;%ii的初始值，代表属性的列号
while ii<=size(data,2)
     %计算中心点   
     num=max(unique(target(:,1)));
     point=struct('data',[]);%中心点
     point(num).data=[];
     point(num).target=[];
     for ni=1:size(data,1)
         point(target(ni,1)).data=[point(target(ni,1)).data;data(ni,:)];
     end
     %消除point结构体中的空元素
     i=1;
     while i<=size(point,2)
           if isempty(point(i).data)==1 
              point(i)=[];
           else
              i=i+1;
           end
     end
     center=[];%保存中心点
     for si=1:size(point,2)%求平均中心点
         temp=mean(point(si).data);
         center=[center;temp];%获得中心点
     end
     kresult=kmeans(data,klabel,'emptyaction','drop','start',center);%利用k-means算法进行聚类
     
     %确定聚类结果对应的真实意义
     kweight=zeros(klabel,num);
     for j=1:size(kresult,1)
         kweight(kresult(j,1),target(j,1))=kweight(kresult(j,1),target(j,1))+1;
     end
     [c,d]=max(kweight,[],2);%c代表每行的最大值，d代表每行对应的类标签
     %把聚类的类标签改写为实际对应的类标签
     for it=1:size(kresult,1)
         kresult(it,1)=d(kresult(it,1),1);
     end
     %聚类结果的正确率
     kcount=0;
     for tt=1:size(kresult,1)
         if kresult(tt,1)==target(tt,1)
             kcount=kcount+1;
         end
     end
     right0=kcount/(size(kresult,1));
     
     %删除第ii维属性，相当于约简
     center(:,ii)=[];%%删除第ii维属性属性后的数据
     datai=data;%防止原来数据遭到破坏
     datai(:,ii)=[];%删除第ii维属性属性后的数据
     kresult2=kmeans(datai,klabel,'emptyaction','drop','start',center);
     %确定聚类结果对应的真实类标签
     kweight2=zeros(klabel,num);
     for ai=1:size(kweight2,1)
         kweight2(kresult2(ai,1),target(ai,1))=kweight2(kresult2(ai,1),target(ai,1))+1;
     end
     [c,d2]=max(kweight2,[],2);%采用多数投票的方式决定最后的类标签，c代表最大值，d2代表最大值对应的类标签
     %修改聚类结果为对应的真实类标签
     for yi=1:size(d2,1)
         kresult2(yi,1)=d2(kresult2(yi,1),1);
     end
     %统计正确率
     kcount2=0;
     for yj=1:size(kresult2,1)
         if kresult2(yj,1)==target(yj,1)
             kcount2=kcount2+1;
         end
     end
     right2=kcount2/(size(target,1));%计算正确率
     
     %判断属性是否可以约简     
     if right2>=right0 %说明当前属性可以约简
         data(:,ii)=[];
     else%说明当前属性不可约简
         ii=ii+1;
     end
     
end %endwhile  ii
data1=data;

end %end function

