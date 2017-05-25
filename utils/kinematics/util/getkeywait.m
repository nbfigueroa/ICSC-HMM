function ch = getkeywait(m) 

% GETKEYWAIT - get a key 
%   CH = GETKEYWAIT(P) waits for a keypress for a maximum of P seconds. It returns a
%   keypress as an ascii number, including backspace (8), space (32), enter (13),
%   etc. CH is a double. P should be a positive number.
%   If no key is pressed within P seconds, -1 is returned. If something
%   went wrong during an empty matrix is returned.
%
%  See also INPUT, GETKEY (FileExchange)

% 2005 Jos
% Feel free to (ab)use, modify or change this contribution

error(nargchk(1,1,nargin)) ;
callstr = ['set(gcbf,''Userdata'',double(get(gcbf,''Currentcharacter''))) ; uiresume '] ;
tt = [] ;

if isnumeric(m),
    m = m(1) ;
    if m > 0,
        tt = timer ;
        tt.timerfcn = 'uiresume' ;
        tt.startdelay = m ;            
    else
        error('Argument should be positive.') ;
    end
else
    error('Argument should be a single positive number.') ;
end

% Set up the figure
% May be the position property should be individually tweaked to avoid visibility
fh = figure('keypressfcn',callstr, ...
    'windowstyle','modal',...    
    'position',[0 0 1 1],...
    'Name','GETKEYWAIT', ...
    'userdata',-1) ; 
try
    % Wait for something to happen or the timer to run out
    start(tt) ;    
    uiwait ;
    ch = get(fh,'Userdata') ;
catch
    % Something went wrong, return and empty matrix.
    ch = [] ;
end

stop(tt) ;
delete(tt) ; 

close(fh) ;
