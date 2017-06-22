%% Load Aligned Dataset
data_path = './test-data/';
load(strcat(data_path,'Rolling/proc-data-labeled-aligned.mat')) 

%% Rotate Time-Series
offset1 = [0.45 -0.45 zeros(1,11)]';
Rz = rotz(-pi/3);
clear data
for d=2:3:length(Data)
    data = Data{d} + repmat(offset1,[1 length(Data{d})]);
        for dd=1:length(data)
            data(1:3,dd) = Rz*data(1:3,dd);
            data(8:10,dd) = Rz*data(8:10,dd);
        end         
    Data{d} = data - repmat(offset1,[1 length(Data{d})]);
end