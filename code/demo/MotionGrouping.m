function [groups] = MotionGrouping(diff, thres)
for i=1:length(diff)
    tmp_groups{i,:} = find(diff(i,:)<thres);
end

i = 1;
tmp_groups_rest = tmp_groups;
clear groups
while(~isempty(tmp_groups_rest))
    group = tmp_groups_rest{1,:};
    groups{i,:}=group;
    tmp_groups_rest{1} = [];
    tmp_groups_rest = tmp_groups_rest(~cellfun(@isempty, tmp_groups_rest));
    
    for l=1:length(tmp_groups_rest)
        if sum(ismember(tmp_groups_rest{l,:},groups{i,:}))>0
            groups_tmp = [groups{i,:} tmp_groups_rest{l,:}];
            groups{i,:} = unique(groups_tmp);
            tmp_groups_rest{l,:}=[];
        end
    end
    tmp_groups_rest = tmp_groups_rest(~cellfun(@isempty, tmp_groups_rest));
    i=i+1;
end
end
