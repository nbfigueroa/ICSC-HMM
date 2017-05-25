function [ko, k] = valid_findk(S, kfind, id, k, N)

  if kfind(id) == 2
    [high, ko] = max(S);
  elseif kfind(id) == 1
    [low, ko] = min(S);
  elseif kfind(id) == 5
    ke = find(S<=10);
    if ~isempty(ke)
      ko = ke(1);
    else
      ko = [];
    end
    k = k-1;
  elseif kfind(id) == 3 || kfind(id) == 4
    high = [S(1) S(1:N-1)];
    low = [S(2:N) S(N)];
    ke = high+low-2*S;  % second differences
    if kfind(id) == 3
      [low, ko] = min(ke);
    else
      [low, ko] = max(ke);
    end
  elseif kfind(id) < 0
    ko = -kfind(id);
  elseif kfind(id) > 9
    ko = [];
  end
