function [ action_sequences ] = extractActionSequences(inf_action_sequence, uni_clust_results, Xn_seg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

action_sequences = [];
k = 0;
for ii=1:length(Xn_seg)
    unsegmented_ts = Xn_seg{ii};
    ts_action_assign = uni_clust_results{ii};
    
    % Find the ordered sequence withing time-series
    ind=strfind(reshape(ts_action_assign(:,1),1,[]),inf_action_sequence);
    
    % Number of Repeated Sequences
    n_seq = length(ind);
    n_act = length(inf_action_sequence);
        
    % Extract Sequences    
    for jj=1:n_seq
        k = k + 1;
        if ind(jj) == 1        
            start_seq = 1;            
        else
            start_seq =  ts_action_assign(ind(jj)-1,2);
        end
        end_seq   = ts_action_assign(ind(jj)+n_act-1,2);        
                
        act_assign = ts_action_assign(ind(jj):ind(jj)+n_act-1,:);
        one_sequence_in_ts = unsegmented_ts(:,start_seq:end_seq);
        seq_offset = act_assign(end,2) - length(one_sequence_in_ts);
        act_assign(:,2) = act_assign(:,2) - seq_offset;
        act_segms = [];
        for kk=1:length(inf_action_sequence)
            if kk==1
                start_action = 1;
            else
                start_action = act_assign(kk-1,2);
            end
            act_segms = [act_segms; start_action act_assign(kk,2)];
        end
        
        act_labels = [];
        act_labels = [act_labels inf_action_sequence(1)];
        for kk=1:length(inf_action_sequence)
            act_labels = [act_labels ones(1,act_segms(kk,2)-act_segms(kk,1))*inf_action_sequence(kk)];
        end
        one_sequence_in_ts = [one_sequence_in_ts;act_labels];
        action_sequences{k,1} = one_sequence_in_ts;        
    end
end

end

