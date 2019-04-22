function [ M,ANNOTD,ATRTIMED ] = rdecg( PATH, filename,samples2read )
% 读取ECG信号，返回信号向量，专家标记，专家标记位置
% 

HEADERFILE= strcat(filename,'.hea');      
ATRFILE= strcat(filename,'.atr');         
DATAFILE=strcat(filename,'.dat');         
SAMPLES2READ=samples2read;          
                          


fprintf(1,'\\n$> WORKING ON %s ...\n', HEADERFILE); 
signalh= fullfile(PATH, HEADERFILE);    
fid1=fopen(signalh,'r');    
z= fgetl(fid1);             
A= sscanf(z, '%*s %d %d %d',[1,3]); 
nosig= A(1);    
sfreq=A(2);     
clear A;        
for k=1:nosig         
    z= fgetl(fid1);
    A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
    dformat(k)= A(1);           
    gain(k)= A(2);              
    bitres(k)= A(3);           
    zerovalue(k)= A(4);         
    firstvalue(k)= A(5);        
end
fclose(fid1);
clear A;

if dformat~= [212,212], error('this script does not apply binary formats different to 212.'); end
signald= fullfile(PATH, DATAFILE);           
fid2=fopen(signald,'r');
A= fread(fid2, [3, SAMPLES2READ], 'uint8')';  
fclose(fid2);
M2H= bitshift(A(:,2), -4);        
M1H= bitand(A(:,2), 15);          
PRL=bitshift(bitand(A(:,2),8),9);     
PRR=bitshift(bitand(A(:,2),128),5);   
M( : , 1)= bitshift(M1H,8)+ A(:,1)-PRL;
M( : , 2)= bitshift(M2H,8)+ A(:,3)-PRR;
if M(1,:) ~= firstvalue, error('inconsistency in the first bit values'); end
switch nosig
case 2
    M( : , 1)= (M( : , 1)- zerovalue(1))/gain(1);
    M( : , 2)= (M( : , 2)- zerovalue(2))/gain(2);
    TIME=(0:(SAMPLES2READ-1))/sfreq;
case 1
    M( : , 1)= (M( : , 1)- zerovalue(1));
    M( : , 2)= (M( : , 2)- zerovalue(1));
    M=M';
    M(1)=[];
    sM=size(M);
    sM=sM(2)+1;
    M(sM)=0;
    M=M';
    M=M/gain(1);
    TIME=(0:2*(SAMPLES2READ)-1)/sfreq;
otherwise  
    disp('Sorting algorithm for more than 2 signals not programmed yet!');
end
clear A M1H M2H PRR PRL;
fprintf(1,'\\n$> LOADING DATA FINISHED \n');
atrd= fullfile(PATH, ATRFILE);      
fid3=fopen(atrd,'r');
A= fread(fid3, [2, inf], 'uint8')';
fclose(fid3);
ATRTIME=[];
ANNOT=[];
sa=size(A);
saa=sa(1);
i=1;
while i<=saa
    annoth=bitshift(A(i,2),-2);
    if annoth==59
        ANNOT=[ANNOT;bitshift(A(i+3,2),-2)];
        ATRTIME=[ATRTIME;A(i+2,1)+bitshift(A(i+2,2),8)+...
                bitshift(A(i+1,1),16)+bitshift(A(i+1,2),24)];
        i=i+3;
    elseif annoth==60
        % nothing to do!
    elseif annoth==61
        % nothing to do!
    elseif annoth==62
        % nothing to do!
    elseif annoth==63
        hilfe=bitshift(bitand(A(i,2),3),8)+A(i,1);
        hilfe=hilfe+mod(hilfe,2);
        i=i+hilfe/2;
    else
        ATRTIME=[ATRTIME;bitshift(bitand(A(i,2),3),8)+A(i,1)];
        ANNOT=[ANNOT;bitshift(A(i,2),-2)];
    end
   i=i+1;
end
ANNOT(length(ANNOT))=[];       
ATRTIME(length(ATRTIME))=[];  
clear A;
ATRTIME= (cumsum(ATRTIME))/sfreq;
ind= find(ATRTIME <= TIME(end));
ATRTIMED= ATRTIME(ind);
ANNOT=round(ANNOT);
ANNOTD= ANNOT(ind);

% -------------------------------------------------------------------------
fprintf(1,'\\n$> ALL FINISHED \n');
end

