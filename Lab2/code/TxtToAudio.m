function []=TxtToAudio(input_file, wav_file)
data=load(input_file);
A=zeros(1, 8000*0.3*size(data,1));
for it=1:4
    disp(it)
    frs=data(:,it);
    Fs=8000;
    Ts=1/Fs;
    t=0:Ts:0.29999;
    for i=1:size(frs,1)
         F_A=2^((frs(i,1)-49)/12)*440;
         A((i-1)*8000*0.3+1:i*8000*0.3)=A((i-1)*8000*0.3+1:i*8000*0.3)+0.25*sin(2*pi*F_A*t);
    end
end
audiowrite(wav_file, A,8000)
plot(A(1:80000))