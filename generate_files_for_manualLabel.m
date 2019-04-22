% 生成用于人工标记的文件，注意修改路径 
%%
close all;clear;clc;
namet=[119];                                          % 要处理的记录
fs=360;
PATH='D:\Program Files (x86)\MATLAB\R2011a\MIT-BIH\'; % 存放原始数据的路径
save_path='E:\ECG_FRCNN\ECG_Segs\';                   % 存放生成文件的路径        
samples2read=650000;
shift_per_step=2*fs;                                  % 每次向前移动2s，即片段与片段之间重叠3s
seg_length=5*fs;                                      % 切取5s片段 
for i=1:length(namet)
    filename=num2str(namet(i));
    [M,ANNOTD,ATRTIMED]=rdecg(PATH,filename,samples2read);
    mkdir(strcat(save_path,filename,'_MASK'));
    sig=M(:,1);
    ann=ANNOTD;
    atrt=round(ATRTIMED*fs);
    start_seg=1;
    No_seg=1;
    while start_seg + seg_length-1 <= samples2read
        pL=start_seg;pR=start_seg+seg_length-1;
        seg_t=sig(pL:pR);
        select=(atrt>=pL) &(atrt<=pR);
        if sum(select)==0
            start_seg=start_seg+shift_per_step;
            No_seg=No_seg+1;
            close all;
            continue;
        end
        [~,indt]=max(select);aL=indt;
        [~,indt]=max(flip(select));aR=length(ann)-indt+1;
        ann_seg=ann(aL:aR);
        ann_pos=atrt(aL:aR)-pL+1;
        plot(seg_t,'k');axis([1,seg_length,min(seg_t)-0.2,max(seg_t)+0.2]);
        hold on;
        
        for k=1:length(ann_seg)
            switch ann_seg(k)
                % 根据专家标记在图上标注相应类型
                case 1
                    plot(ann_pos(k),seg_t(ann_pos(k)),'ro','markersize',15);
                case 2
                    plot(ann_pos(k),seg_t(ann_pos(k)),'b*','markersize',15);
                case 3
                    plot(ann_pos(k),seg_t(ann_pos(k)),'g+','markersize',15);
                case 5
                    plot(ann_pos(k),seg_t(ann_pos(k)),'ms','markersize',15);
                otherwise
                    plot(ann_pos(k),seg_t(ann_pos(k)),'kx','markersize',15);
                % 这里写了四类，如果有需要也可以在下面增加其他类型
                % case X
                %   plot(ann_pos(k),seg_t(ann_pos(k)),'a_style','markersize',15);
                % case Y
                % ...
                
            end
        end
        
        set(gcf,'outerposition',get(0,'screensize'));
        set(gca,'position',[0 0 1 1]);axis off;
        savename_fig=strcat(save_path,filename,'_MASK','\seg',num2str(No_seg),'_',filename,'.png');
        saveas(gcf,savename_fig); % 保存当前片段图像
        savename_mat=strcat(save_path,filename,'_MASK','\seg',num2str(No_seg),'_',filename,'.mat'); 
        save(savename_mat,'seg_t'); % 保存信号片段
        savename_ann=strcat(save_path,filename,'_MASK','\ann',num2str(No_seg),'_',filename,'.mat');
        save(savename_ann,'ann_pos'); % 保存相应标记位置
        start_seg=start_seg+shift_per_step;
        No_seg=No_seg+1;
        close all;
    end
end