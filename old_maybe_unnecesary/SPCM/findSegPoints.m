function [seg_results] = findSegPoints(Zest)
    curr = Zest(1);
    seg_results = [];
    for i=1:length(Zest)
        if (Zest(i)~=curr || i==length(Zest))
            time = i-1;
            if (i==length(Zest))
                time = i;
            end
            seg_point = [curr time];  
            seg_results = [seg_results; seg_point];
            curr = Zest(i);
        end
    end
    
end